# -*- coding: utf-8 -*-
"""
tabular_infer_live.py (final)
以 sklearn (data/tabular_baseline.pkl) 即時推論「理解/困惑」。
- 先用 FaceDetection 取臉框
- Landmark 依序嘗試：臉框 → 放大臉框 → 全圖 →（若非Tasks）靜態模式
- 一旦取得 landmarks，直接線性縮放到 224x224 空間計算特徵與推論（不再第二次偵測）
- 內建中文字型搜尋，畫面 HUD 含 debug 旗標（FD/LM1~LM4）

需求：
  pip install opencv-python mediapipe pillow joblib numpy
"""

import time, cv2, joblib, numpy as np, mediapipe as mp
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image

# ====== 基本設定 ======
MODEL_PKL   = Path("data/tabular_baseline.pkl")
TASK_MODEL  = Path("models/face_landmarker.task")   # 若存在則優先用 MediaPipe Tasks
CAM_INDEX   = 0
FRAME_W, FRAME_H = 640, 480
DRAW_LANDMARKS   = False  # True 會開 mesh-debug 視窗顯示綠點

# ====== 中文字型 ======
def _try_font(paths, size):
    for p in paths:
        p = Path(p)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return None

def get_zh_font(size=22):
    search = [
        Path("fonts/msjh.ttc"), Path("fonts/NotoSansTC-Regular.otf"),
        Path(r"C:\Windows\Fonts\msjh.ttc"), Path(r"C:\Windows\Fonts\msjhbd.ttc"), Path(r"C:\Windows\Fonts\mingliu.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"), Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansTC-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    return _try_font(search, size)

ZH_FONT       = get_zh_font(22) or ImageFont.load_default()
ZH_FONT_SMALL = get_zh_font(18) or ImageFont.load_default()
CHINESE_OK    = get_zh_font(22) is not None

def put_text(img_bgr, text, xy, font=None, fill=(0,255,0), outline=(0,0,0), outline_w=2):
    font = font or ZH_FONT
    if not CHINESE_OK:
        try: text = text.encode("ascii","ignore").decode("ascii")
        except Exception: text = "[no-zh-font]"
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    x, y = xy
    if outline_w and outline is not None:
        for dx in range(-outline_w, outline_w+1):
            for dy in range(-outline_w, outline_w+1):
                if dx==0 and dy==0: continue
                draw.text((x+dx, y+dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ====== Face Mesh 索引與幾何 ======
L_OUT,L_IN,L_UP,L_DN = 33,133,159,145
R_OUT,R_IN,R_UP,R_DN = 263,362,386,374
M_LEFT,M_RIGHT,M_UP,M_DN = 61,291,13,14
LBROW = [70,63,105]; RBROW = [336,296,334]

def dist(a,b): a,b=np.asarray(a,float),np.asarray(b,float); return float(np.linalg.norm(a-b))
def EAR(pts,u,d,o,i): v=abs(pts[u][1]-pts[d][1]); h=abs(pts[o][0]-pts[i][0])+1e-6; return v/h
def MAR(pts):
    up,dn,lf,rt = pts[M_UP],pts[M_DN],pts[M_LEFT],pts[M_RIGHT]
    return abs(up[1]-dn[1])/(abs(lf[0]-rt[0])+1e-6)
def mouth_curve_M(pts):
    lf,rt,up,dn = pts[M_LEFT],pts[M_RIGHT],pts[M_UP],pts[M_DN]
    return -(((up[1]+dn[1])/2)-((lf[1]+rt[1])/2))
def brow_raise_B(pts,cl,cr):
    lb,rb = np.mean([pts[i] for i in LBROW],axis=0), np.mean([pts[i] for i in RBROW],axis=0)
    return ((cl[1]-lb[1])+(cr[1]-rb[1]))/2.0

def compute_feats_from_pts(pts224, calib):
    """輸入：224x224 空間的 landmarks 清單 → 回傳特徵 dict"""
    eye_c_l=((pts224[L_OUT][0]+pts224[L_IN][0])/2.0,(pts224[L_UP][1]+pts224[L_DN][1])/2.0)
    eye_c_r=((pts224[R_OUT][0]+pts224[R_IN][0])/2.0,(pts224[R_UP][1]+pts224[R_DN][1])/2.0)
    ipd=dist(eye_c_l,eye_c_r)
    ear_l,ear_r=EAR(pts224,L_UP,L_DN,L_OUT,L_IN),EAR(pts224,R_UP,R_DN,R_OUT,R_IN)
    mar=MAR(pts224); face_area=224.0*224.0
    B_raw=brow_raise_B(pts224,eye_c_l,eye_c_r); M_raw=mouth_curve_M(pts224)
    def upd(k,v):
        mm=calib.setdefault(k,{"min":float(v),"max":float(v)})
        mm["min"]=min(mm["min"],v); mm["max"]=max(mm["max"],v)
    for k,v in [("face_area",face_area),("ipd",ipd),("mar",mar),("B_raw",B_raw),("M_raw",M_raw)]: upd(k,v)
    def minmax(x,lo,hi): 
        if hi-lo<1e-6: return 0.5
        return float(np.clip((x-lo)/(hi-lo),0,1))
    def center_pm1(x,lo,hi):
        if hi-lo<1e-6: return 0.0
        return float(np.clip((x-(lo+hi)/2)/((hi-lo)/2),-1,1))
    Aw=1.0-minmax(face_area,calib["face_area"]["min"],calib["face_area"]["max"])
    Ca=      minmax(ipd,      calib["ipd"]["min"],      calib["ipd"]["max"])
    Hp=      minmax(M_raw,    calib["M_raw"]["min"],    calib["M_raw"]["max"])
    B=center_pm1(B_raw,calib["B_raw"]["min"],calib["B_raw"]["max"])
    M=center_pm1(M_raw,calib["M_raw"]["min"],calib["M_raw"]["max"])
    mag=(B*B+M*M)**0.5; ang=float(np.arctan2(B,M))
    feats={"Aw":Aw,"Ca":Ca,"Hp":Hp,"B":B,"M":M,"magnitude":mag,"direction":ang,
           "ear_l":ear_l,"ear_r":ear_r,"mar":mar,"ipd":ipd,"face_area":face_area}
    return feats

def ensure_feature_order(feats_dict, feat_names):
    vec,missing=[],[]
    for k in feat_names:
        if k in feats_dict: vec.append(float(feats_dict[k]))
        else: vec.append(0.0); missing.append(k)
    return np.array([vec],dtype=float), missing

# ====== 亮度補償（提昇低光穩定度）=======
def enhance_light(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    return cv2.cvtColor(cv2.merge([y,cr,cb]), cv2.COLOR_YCrCb2BGR)

# ====== Landmarker：Tasks 優先，否則用 solutions.face_mesh ======
class Landmarker:
    def __init__(self):
        self.use_tasks = TASK_MODEL.exists()
        if self.use_tasks:
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision
                BaseOptions = mp_python.BaseOptions
                VisionRunningMode = vision.RunningMode
                FaceLandmarker = vision.FaceLandmarker
                FaceLandmarkerOptions = vision.FaceLandmarkerOptions
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(TASK_MODEL)),
                    running_mode=VisionRunningMode.VIDEO,
                    num_faces=1,
                    output_facial_transformation_matrixes=False
                )
                self.landmarker = FaceLandmarker.create_from_options(options)
                self._tasks_ts_ms = 0
                print("✅ 使用 MediaPipe Tasks FaceLandmarker")
            except Exception as e:
                print("⚠️ 無法載入 Tasks，改用 solutions.face_mesh。原因：", e)
                self.use_tasks = False
        if not self.use_tasks:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, refine_landmarks=True,
                max_num_faces=1, min_detection_confidence=0.3, min_tracking_confidence=0.3
            )
            print("✅ 使用 solutions.face_mesh")

    def landmarks(self, img_bgr):
        """回傳像素座標 list[(x,y)] 或 None"""
        if self.use_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            self._tasks_ts_ms += 33
            res = self.landmarker.detect_for_video(mp_image, self._tasks_ts_ms)
            if res.face_landmarks and len(res.face_landmarks) > 0:
                h, w = img_bgr.shape[:2]
                return [(lm.x*w, lm.y*h) for lm in res.face_landmarks[0]]
            return None
        else:
            r = self.face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if not r.multi_face_landmarks: return None
            h, w = img_bgr.shape[:2]
            return [(p.x*w, p.y*h) for p in r.multi_face_landmarks[0].landmark]

# ====== 主程式 ======
def main():
    if not MODEL_PKL.exists():
        print("❌ 找不到模型：", MODEL_PKL); return
    bundle = joblib.load(MODEL_PKL)
    scaler, clf, FEATS = bundle["scaler"], bundle["clf"], bundle["feats"]
    print("✅ 已載入分類模型，特徵：", FEATS)
    if not CHINESE_OK:
        print("⚠️ 未找到中文字型，畫面將以英數字顯示（可把字型放到 ./fonts/）。")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_H)
    if not cap.isOpened(): print("❌ 無法開啟攝影機"); return

    det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    lm  = Landmarker()

    calib = {}
    cv2.namedWindow("tabular-infer", cv2.WINDOW_NORMAL)
    t0, nframe, missing_warned = time.time(), 0, False

    while True:
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((FRAME_H,FRAME_W,3),np.uint8)
            frame = put_text(frame,"Camera read fail...",(10,20),font=ZH_FONT,fill=(0,0,255))
            cv2.imshow("tabular-infer",frame)
            if (cv2.waitKey(1)&0xFF)==27: break
            continue

        frame = enhance_light(frame)
        h,w = frame.shape[:2]
        pred_str, prob_str = "未偵測到臉", ""
        flags = {"FD":0,"LM1":0,"LM2":0,"LM3":0,"LM4":0}

        # 臉偵測
        res = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.detections:
            flags["FD"]=1
            d = res.detections[0].location_data.relative_bounding_box
            x1 = max(0, int(d.xmin*w)); y1 = max(0, int(d.ymin*h))
            ww = int(d.width*w); hh = int(d.height*h)
            x2 = min(w-1, x1+ww); y2 = min(h-1, y1+hh)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
            face = frame[y1:y2, x1:x2].copy()

            pts = None

            # Try-1：臉框
            if face.size > 0:
                pts = lm.landmarks(face)
                if pts is not None: flags["LM1"]=1

            # Try-2：放大臉框
            if pts is None and face.size > 0:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                bw, bh = int(ww*0.8), int(hh*0.8)
                nx1, ny1 = max(0, cx-bw), max(0, cy-bh)
                nx2, ny2 = min(w-1, cx+bw), min(h-1, cy+bh)
                big = frame[ny1:ny2, nx1:nx2].copy()
                big = cv2.copyMakeBorder(big, 20,20,20,20, cv2.BORDER_REFLECT)
                pts_big = lm.landmarks(big)
                if pts_big is not None:
                    flags["LM2"]=1
                    offx, offy = (big.shape[1]-(nx2-nx1))//2, (big.shape[0]-(ny2-ny1))//2
                    pts = [(px-offx, py-offy) for (px,py) in pts_big]

            # Try-3：全圖
            if pts is None:
                pts_full = lm.landmarks(frame)
                if pts_full is not None:
                    flags["LM3"]=1
                    pts = [(px-x1, py-y1) for (px,py) in pts_full]

            # Try-4：靜態模式（僅 solutions 可用）
            if pts is None and not lm.use_tasks:
                mesh_static = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, refine_landmarks=True, max_num_faces=1,
                    min_detection_confidence=0.3, min_tracking_confidence=0.3
                )
                r = mesh_static.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if r.multi_face_landmarks:
                    flags["LM4"]=1
                    pts = [(p.x*w - x1, p.y*h - y1) for p in r.multi_face_landmarks[0].landmark]

            # 一旦有 landmarks → 直接縮放到 224 空間並推論
            if pts is not None and face.size > 0:
                fh, fw = face.shape[:2]
                sx, sy = 224.0/max(1,fw), 224.0/max(1,fh)
                pts224 = [(px*sx, py*sy) for (px,py) in pts]

                if DRAW_LANDMARKS:
                    dbg = face.copy()
                    for (px,py) in pts:
                        cv2.circle(dbg, (int(px),int(py)), 1, (0,255,0), -1)
                    cv2.imshow("mesh-debug", dbg)

                feats = compute_feats_from_pts(pts224, calib)
                x, missing = ensure_feature_order(feats, FEATS)
                if missing and not missing_warned:
                    print("⚠️ 推論缺少特徵（以 0.0 代入）：", missing)
                    missing_warned = True
                xs = scaler.transform(x)
                if hasattr(clf,"predict_proba"):
                    p_conf=float(clf.predict_proba(xs)[0][1])
                elif hasattr(clf,"decision_function"):
                    z=float(clf.decision_function(xs)[0]); p_conf=1/(1+np.exp(-z))
                else:
                    p_conf=float(clf.predict(xs)[0])
                pred_str = "困惑" if p_conf>=0.5 else "理解"
                prob_str = f"conf:{p_conf:.2f}"

        # HUD
        nframe += 1; fps = nframe / max(1e-6,(time.time()-t0))
        flag_txt = " ".join([f"{k}:{v}" for k,v in flags.items()])
        hud = f"{'Tasks' if TASK_MODEL.exists() else 'FaceMesh'} | {flag_txt} | {pred_str} {prob_str} | FPS:{fps:.1f} [ESC]"
        frame = put_text(frame, hud, (10, 18), font=ZH_FONT, fill=(0,255,0), outline=(0,0,0), outline_w=2)
        cv2.imshow("tabular-infer", frame)
        if (cv2.waitKey(1)&0xFF)==27: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

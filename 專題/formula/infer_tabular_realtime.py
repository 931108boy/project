# -*- coding: utf-8 -*-
"""
tabular_infer_live.py  (FULL)
以 sklearn (data/tabular_baseline.pkl) 做即時「理解/困惑」推論。
需求：
  pip install opencv-python mediapipe pillow joblib numpy
確保安裝的是 opencv-python (非 headless)。

操作：
  視窗左上顯示 臉數 / 預測 / 置信度 / FPS / 提示
  [ESC] 退出
"""

import time, cv2, joblib, numpy as np, mediapipe as mp
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image

# ========== 參數 ==========
MODEL_PATH = Path("data/tabular_baseline.pkl")
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
DRAW_LANDMARKS = False  # 想看臉上綠點就改 True

# ========== 中文字型工具 ==========
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
        Path("fonts/msjh.ttc"),
        Path("fonts/NotoSansTC-Regular.otf"),
        # Windows
        Path(r"C:\Windows\Fonts\msjh.ttc"),
        Path(r"C:\Windows\Fonts\msjhbd.ttc"),
        Path(r"C:\Windows\Fonts\mingliu.ttc"),
        # macOS
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        # Linux (Noto)
        Path("/usr/share/fonts/opentype/noto/NotoSansTC-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    return _try_font(search, size)

ZH_FONT       = get_zh_font(22) or ImageFont.load_default()
ZH_FONT_SMALL = get_zh_font(18) or ImageFont.load_default()
CHINESE_OK    = get_zh_font(22) is not None

def put_text(img_bgr, text, xy, font=None, fill=(0,255,0), outline=(0,0,0), outline_w=2):
    """在 OpenCV 影像上以 Pillow 畫字（支援中文、外框）"""
    font = font or ZH_FONT
    if not CHINESE_OK:
        try:
            text = text.encode("ascii", "ignore").decode("ascii")
        except Exception:
            text = "[no-zh-font]"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    x, y = xy
    if outline_w and outline is not None:
        for dx in range(-outline_w, outline_w+1):
            for dy in range(-outline_w, outline_w+1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x+dx, y+dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ========== 幾何/特徵工具 ==========
# MediaPipe Face Mesh 索引
L_OUT,L_IN,L_UP,L_DN = 33,133,159,145
R_OUT,R_IN,R_UP,R_DN = 263,362,386,374
M_LEFT,M_RIGHT,M_UP,M_DN = 61,291,13,14
LBROW = [70,63,105]
RBROW = [336,296,334]

def dist(a,b):
    a,b = np.asarray(a,float), np.asarray(b,float)
    return float(np.linalg.norm(a-b))

def EAR(pts, up_idx, dn_idx, out_idx, in_idx):
    v = abs(pts[up_idx][1]-pts[dn_idx][1])
    h = abs(pts[out_idx][0]-pts[in_idx][0]) + 1e-6
    return v/h

def MAR(pts):
    up, dn = pts[M_UP], pts[M_DN]
    left, right = pts[M_LEFT], pts[M_RIGHT]
    h = abs(up[1]-dn[1])
    w = abs(left[0]-right[0]) + 1e-6
    return h/w

def mouth_curve_M(pts):
    left, right = pts[M_LEFT], pts[M_RIGHT]
    up, dn = pts[M_UP], pts[M_DN]
    corners_y = (left[1] + right[1]) / 2.0
    mid_y = (up[1] + dn[1]) / 2.0
    return -(mid_y - corners_y)  # 正：上揚；負：下垂

def brow_raise_B(pts, eye_c_l, eye_c_r):
    lb = np.mean([pts[i] for i in LBROW], axis=0)
    rb = np.mean([pts[i] for i in RBROW], axis=0)
    dl = eye_c_l[1] - lb[1]
    dr = eye_c_r[1] - rb[1]
    return (dl + dr) / 2.0

def affine_align_by_eyes(img, left_eye, right_eye, out_size=224, eye_pos=(0.35, 0.4)):
    dst = np.float32([
        [eye_pos[0]*out_size, eye_pos[1]*out_size],
        [(1-eye_pos[0])*out_size, eye_pos[1]*out_size],
        [eye_pos[0]*out_size, (eye_pos[1]+0.4)*out_size],
    ])
    mid = ((left_eye[0]+right_eye[0])/2, (left_eye[1]+right_eye[1])/2)
    down = (mid[0], mid[1] + (right_eye[0]-left_eye[0])*0.9)
    src = np.float32([left_eye, right_eye, down])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def compute_feats_from_aligned(aligned, mesh, calib):
    """回傳 feats, calib；若本影格無法取到 mesh 則回 None, calib。"""
    r = mesh.process(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    if not r.multi_face_landmarks:
        return None, calib
    pts = [(p.x*224, p.y*224) for p in r.multi_face_landmarks[0].landmark]

    eye_c_l = ((pts[L_OUT][0]+pts[L_IN][0])/2.0, (pts[L_UP][1]+pts[L_DN][1])/2.0)
    eye_c_r = ((pts[R_OUT][0]+pts[R_IN][0])/2.0, (pts[R_UP][1]+pts[R_DN][1])/2.0)
    ipd = dist(eye_c_l, eye_c_r)
    ear_l = EAR(pts, L_UP, L_DN, L_OUT, L_IN)
    ear_r = EAR(pts, R_UP, R_DN, R_OUT, R_IN)
    mar = MAR(pts)
    face_area = 224.0*224.0
    B_raw = brow_raise_B(pts, eye_c_l, eye_c_r)
    M_raw = mouth_curve_M(pts)

    # 線上校正：更新 min/max
    def upd(key, v):
        mm = calib.setdefault(key, {"min": float(v), "max": float(v)})
        mm["min"] = float(min(mm["min"], v)); mm["max"] = float(max(mm["max"], v))
    for k,v in [("face_area",face_area),("ipd",ipd),("mar",mar),("B_raw",B_raw),("M_raw",M_raw)]: upd(k,v)

    def minmax_norm(x, lo, hi, eps=1e-6):
        if hi - lo < eps: return 0.5
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    def center_pm1(x, lo, hi):
        if hi-lo<1e-6: return 0.0
        return float(np.clip((x-(lo+hi)/2)/((hi-lo)/2), -1, 1))

    Aw = 1.0 - minmax_norm(face_area, calib["face_area"]["min"], calib["face_area"]["max"])
    Ca =       minmax_norm(ipd,       calib["ipd"]["min"],       calib["ipd"]["max"])
    Hp =       minmax_norm(M_raw,     calib["M_raw"]["min"],     calib["M_raw"]["max"])
    B  = center_pm1(B_raw, calib["B_raw"]["min"], calib["B_raw"]["max"])
    M  = center_pm1(M_raw, calib["M_raw"]["min"], calib["M_raw"]["max"])

    mag = (B*B + M*M) ** 0.5
    ang = float(np.arctan2(B, M))

    feats = {"Aw":Aw,"Ca":Ca,"Hp":Hp,"B":B,"M":M,"magnitude":mag,"direction":ang,
             "ear_l":ear_l,"ear_r":ear_r,"mar":mar,"ipd":ipd,"face_area":face_area}
    return feats, calib

def ensure_feature_order(feats_dict, feat_names):
    """依訓練時的特徵順序產生向量；缺欄用 0.0 並回報一次。"""
    vec, missing = [], []
    for k in feat_names:
        if k in feats_dict:
            vec.append(float(feats_dict[k]))
        else:
            vec.append(0.0)
            missing.append(k)
    return np.array([vec], dtype=float), missing

# ========== 主程式 ==========
def main():
    if not MODEL_PATH.exists():
        print("❌ 找不到模型：", MODEL_PATH)
        return
    bundle = joblib.load(MODEL_PATH)
    scaler, clf, FEATS = bundle["scaler"], bundle["clf"], bundle["feats"]
    print("✅ 已載入模型，特徵：", FEATS)
    if not CHINESE_OK:
        print("⚠️ 未找到中文字型，畫面將以英數字顯示。可把字型放到 ./fonts/ 內。")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return

    # 近距離臉更穩：model_selection=0
    det  = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True,
                                           max_num_faces=1, min_detection_confidence=0.5)

    calib = {}
    cv2.namedWindow("tabular-infer", cv2.WINDOW_NORMAL)
    t0, nframe, missing_warned = time.time(), 0, False

    while True:
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
            frame = put_text(frame, "Camera read fail...", (10, 20), font=ZH_FONT, fill=(0,0,255))
            cv2.imshow("tabular-infer", frame)
            if (cv2.waitKey(1)&0xFF)==27: break
            continue

        h, w = frame.shape[:2]
        n_faces, pred_str, prob_str = 0, "未偵測到臉", ""
        warn_line = ""

        # 臉偵測
        res = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.detections:
            n_faces = len(res.detections)
            d = res.detections[0].location_data.relative_bounding_box
            x1 = max(0, int(d.xmin*w)); y1 = max(0, int(d.ymin*h))
            ww = int(d.width*w); hh = int(d.height*h)
            x2 = min(w-1, x1+ww); y2 = min(h-1, y1+hh)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
            face = frame[y1:y2, x1:x2].copy()

            # 先在臉框內跑 FaceMesh
            r1 = mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            used_frame_for_mesh = "face"
            if not r1.multi_face_landmarks:
                # 失敗時：提示 + 改用整張圖再試一次
                warn_line = "⚠️ FaceMesh 失敗（嘗試整張畫面）"
                r1 = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                used_frame_for_mesh = "full"

            if r1.multi_face_landmarks:
                # 取得眼點座標（依據所用的影像空間做換算）
                if used_frame_for_mesh == "face":
                    pts = [(p.x*face.shape[1], p.y*face.shape[0]) for p in r1.multi_face_landmarks[0].landmark]
                    left_eye  = (pts[L_OUT][0], pts[L_UP][1])
                    right_eye = (pts[R_OUT][0], pts[R_UP][1])
                else:  # full
                    pts = [(p.x*w, p.y*h) for p in r1.multi_face_landmarks[0].landmark]
                    left_eye  = (pts[L_OUT][0]-x1, pts[L_UP][1]-y1)
                    right_eye = (pts[R_OUT][0]-x1, pts[R_UP][1]-y1)

                aligned = affine_align_by_eyes(face, left_eye, right_eye, out_size=224)

                # （可選）畫 landmark 以確認
                if DRAW_LANDMARKS:
                    for lm in r1.multi_face_landmarks[0].landmark:
                        cx = int(lm.x*face.shape[1]); cy = int(lm.y*face.shape[0])
                        cv2.circle(face, (cx,cy), 1, (0,255,0), -1)
                    cv2.imshow("mesh-debug", face)

                feats, calib = compute_feats_from_aligned(aligned, mesh, calib)
                if feats is not None:
                    x, missing = ensure_feature_order(feats, FEATS)
                    if missing and not missing_warned:
                        print("⚠️ 推論缺少特徵（以 0.0 代入）：", missing)
                        missing_warned = True
                    xs = scaler.transform(x)
                    if hasattr(clf, "predict_proba"):
                        p_conf = float(clf.predict_proba(xs)[0][1])
                    elif hasattr(clf, "decision_function"):
                        z = float(clf.decision_function(xs)[0]); p_conf = 1/(1+np.exp(-z))
                    else:
                        p_conf = float(clf.predict(xs)[0])
                    pred_str = "困惑" if p_conf >= 0.5 else "理解"
                    prob_str = f"conf:{p_conf:.2f}"
            else:
                warn_line = "⚠️ FaceMesh 無法擷取臉部特徵"

        # HUD（就算無臉/無 landmark 也會顯示）
        nframe += 1
        fps = nframe / max(1e-6, (time.time()-t0))
        hud = f"臉數:{n_faces}  {pred_str} {prob_str}  FPS:{fps:.1f}   [ESC 離開]"
        frame = put_text(frame, hud, (10, 18), font=ZH_FONT, fill=(0,255,0), outline=(0,0,0), outline_w=2)
        if warn_line:
            frame = put_text(frame, warn_line, (10, 44), font=ZH_FONT_SMALL, fill=(0,0,255), outline=(0,0,0), outline_w=2)

        cv2.imshow("tabular-infer", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
資料蒐集 + 特徵計算 + 弱標註（支援中文顯示）
快捷鍵：
  S：開始/停止存檔
  H：顯示/隱藏說明
  ESC：離開
"""

import cv2, time, json, math, csv, argparse, os, sys
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

# ---------- 路徑 ----------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ALIGNED = DATA / "aligned"
CALIB = DATA / "calib"
META = DATA / "meta.csv"

# ---------- 字型載入（支援中文；找不到就退回英文） ----------
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
        ROOT / "fonts" / "msjh.ttc",
        ROOT / "fonts" / "NotoSansTC-Regular.otf",
        # Windows 常見
        "C:/Windows/Fonts/msjh.ttc",
        "C:/Windows/Fonts/msjhbd.ttc",
        "C:/Windows/Fonts/mingliu.ttc",
        # macOS 常見
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/ヒラギノ角ゴシック W3.ttc",  # 有時可顯示中日文
        # Linux 常見（裝了思源黑體時）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansTC-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    f = _try_font(search, size)
    return f

ZH_FONT = get_zh_font(22)
ZH_FONT_SMALL = get_zh_font(18)
CHINESE_OK = ZH_FONT is not None

def put_ch_text(img_bgr, text, xy, font=None, fill=(0,255,0), outline=(0,0,0), outline_w=2):
    """
    在 OpenCV 影像上用 Pillow 畫中文（含外框）
    img_bgr: np.ndarray(BGR)
    text: str
    xy: (x, y)
    """
    if not CHINESE_OK:
        # 找不到中文字型 → 退回英文（簡短顯示避免亂碼）
        try:
            text = text.encode("ascii", "ignore").decode("ascii")
        except Exception:
            text = "[no-zh-font]"
    font = font or ZH_FONT
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    x, y = xy
    if outline_w and outline is not None:
        # 畫外框（八方向）
        for dx in range(-outline_w, outline_w+1):
            for dy in range(-outline_w, outline_w+1):
                if dx == 0 and dy == 0: 
                    continue
                draw.text((x+dx, y+dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------- 小工具 ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path, default=None):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default

def save_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def minmax_norm(x, lo, hi, eps=1e-6):
    if hi - lo < eps: return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def dist(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float(np.linalg.norm(a - b))

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

# ---------- FaceMesh 常用索引 ----------
L_OUT, L_IN, L_UP, L_DN = 33, 133, 159, 145
R_OUT, R_IN, R_UP, R_DN = 263, 362, 386, 374
M_LEFT, M_RIGHT, M_UP, M_DN = 61, 291, 13, 14
LBROW = [70, 63, 105]
RBROW = [336, 296, 334]

def eye_center(pts, out_idx, in_idx, up_idx, dn_idx):
    return ((pts[out_idx][0]+pts[in_idx][0])/2.0, (pts[up_idx][1]+pts[dn_idx][1])/2.0)

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
    raw = (mid_y - corners_y)
    return -raw  # 正：上揚；負：下垂

def brow_raise_B(pts, eye_c_l, eye_c_r):
    lb = np.mean([pts[i] for i in LBROW], axis=0)
    rb = np.mean([pts[i] for i in RBROW], axis=0)
    dl = eye_c_l[1] - lb[1]
    dr = eye_c_r[1] - rb[1]
    return (dl + dr) / 2.0

def compute_features(pts, face_box, calib):
    eye_c_l = eye_center(pts, L_OUT, L_IN, L_UP, L_DN)
    eye_c_r = eye_center(pts, R_OUT, R_IN, R_UP, R_DN)
    ipd = dist(eye_c_l, eye_c_r)
    ear_l = EAR(pts, L_UP, L_DN, L_OUT, L_IN)
    ear_r = EAR(pts, R_UP, R_DN, R_OUT, R_IN)
    mar = MAR(pts)
    x1,y1,x2,y2 = face_box
    face_area = max(1.0, (x2-x1)*(y2-y1))
    B_raw = brow_raise_B(pts, eye_c_l, eye_c_r)
    M_raw = mouth_curve_M(pts)

    def upd(key, v):
        mm = calib.setdefault(key, {"min": float(v), "max": float(v)})
        mm["min"] = float(min(mm["min"], v))
        mm["max"] = float(max(mm["max"], v))
    for k, v in [("face_area", face_area), ("ipd", ipd), ("mar", mar), ("B_raw", B_raw), ("M_raw", M_raw)]:
        upd(k, v)

    Aw = 1.0 - minmax_norm(face_area, calib["face_area"]["min"], calib["face_area"]["max"])
    Ca =       minmax_norm(ipd,       calib["ipd"]["min"],       calib["ipd"]["max"])
    Hp =       minmax_norm(M_raw,     calib["M_raw"]["min"],     calib["M_raw"]["max"])

    def center_to_pm1(x, lo, hi):
        if hi - lo < 1e-6: return 0.0
        z = (x - (lo+hi)/2.0) / ((hi-lo)/2.0)
        return float(np.clip(z, -1.0, 1.0))
    B = center_to_pm1(B_raw, calib["B_raw"]["min"], calib["B_raw"]["max"])
    M = center_to_pm1(M_raw, calib["M_raw"]["min"], calib["M_raw"]["max"])

    magnitude = math.sqrt(B*B + M*M)
    direction = math.atan2(B, M)

    if   M >= 0 and B >= 0: phase = "P1"
    elif M <  0 and B >= 0: phase = "P2"
    elif M <  0 and B <  0: phase = "P3"
    else:                    phase = "P4"

    feats = {
        "face_area": face_area, "ipd": ipd, "ear_l": ear_l, "ear_r": ear_r, "mar": mar,
        "B_raw": B_raw, "M_raw": M_raw, "B": B, "M": M,
        "magnitude": magnitude, "direction": direction,
        "Aw": Aw, "Ca": Ca, "Hp": Hp, "phase": phase
    }
    return feats, calib

def weak_label(Aw, Ca, Hp):
    if (Hp >= 0.60) and (Ca >= 0.50) and (Aw <= 0.60):
        return "understood"
    if (Hp <= 0.40) and ((Ca <= 0.40) or (Aw >= 0.60)):
        return "confused"
    return "uncertain"

def draw_help_cn(img):
    lines = [
        "S：開始/停止存檔    H：切換說明    ESC：離開",
        "指標：迴避度(Aw) 專注度(Ca) 愉悅度(Hp)；Phase 依(眉B, 嘴M)象限",
        "建議：先校準 30~60 秒，再開始存檔"
    ]
    y = 16
    for t in lines:
        img[:] = put_ch_text(img, t, (10, y), font=ZH_FONT_SMALL, fill=(255,255,255), outline=(0,0,0), outline_w=2)
        y += 26
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None, help="受測者ID，例如 s001")
    parser.add_argument("--fps_save", type=float, default=1.0, help="每秒存幾張（預設 1）")
    args = parser.parse_args()

    subject = args.subject or input("輸入受測者ID（例如 s001）：").strip() or "s001"

    ensure_dir(ALIGNED); ensure_dir(CALIB); ensure_dir(DATA)
    calib_p = CALIB / f"{subject}.json"
    calib = load_json(calib_p, default={})

    write_header = not META.exists()
    fcsv = META.open("a", newline="", encoding="utf-8")
    wr = csv.writer(fcsv)
    if write_header:
        wr.writerow([
            "timestamp","path","subject","label","phase",
            "Aw","Ca","Hp","B","M","magnitude","direction",
            "face_area","ipd","ear_l","ear_r","mar"
        ])

    mp_det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟攝影機"); return

    saving = False
    show_help = True
    last_save_t = 0.0
    idx = 0

    print("啟動成功：S 開/關存檔、H 說明、ESC 離開")
    if not CHINESE_OK:
        print("[提醒] 找不到中文字型，畫面上的中文會以英文替代。請把字型檔放到 ./fonts/ 後重啟。")

    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]

        res = mp_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.detections:
            d = res.detections[0].location_data.relative_bounding_box
            x1 = max(0, int(d.xmin*w)); y1 = max(0, int(d.ymin*h))
            ww = int(d.width*w); hh = int(d.height*h)
            x2 = min(w-1, x1+ww); y2 = min(h-1, y1+hh)
            face = frame[y1:y2, x1:x2].copy()

            r1 = mp_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            if r1.multi_face_landmarks:
                lm = r1.multi_face_landmarks[0]
                pts = [(p.x*face.shape[1], p.y*face.shape[0]) for p in lm.landmark]
                left_eye  = (pts[L_OUT][0], pts[L_UP][1])
                right_eye = (pts[R_OUT][0], pts[R_UP][1])
                aligned = affine_align_by_eyes(face, left_eye, right_eye, out_size=224)

                r2 = mp_mesh.process(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
                if r2.multi_face_landmarks:
                    pts2 = [(p.x*224, p.y*224) for p in r2.multi_face_landmarks[0].landmark]
                    feats, calib = compute_features(pts2, (0,0,224,224), calib)

                    Aw, Ca, Hp = feats["Aw"], feats["Ca"], feats["Hp"]
                    B, M = feats["B"], feats["M"]
                    label = weak_label(Aw, Ca, Hp)

                    # 中文疊字
                    txt1 = f"迴避度:{Aw:.2f}  專注度:{Ca:.2f}  愉悅度:{Hp:.2f}  相位:{feats['phase']}"
                    txt2 = f"眉B:{B:+.2f}  嘴M:{M:+.2f}  強度|mag|:{feats['magnitude']:.2f}  方向dir:{feats['direction']:.2f}"
                    aligned = put_ch_text(aligned, txt1, (8, 10), font=ZH_FONT, fill=(0,255,0), outline=(0,0,0), outline_w=2)
                    aligned = put_ch_text(aligned, txt2, (8, 38), font=ZH_FONT_SMALL, fill=(255,255,255), outline=(0,0,0), outline_w=2)
                    aligned = put_ch_text(aligned, f"弱標註：{label}", (8, 66), font=ZH_FONT_SMALL, fill=(255,255,0), outline=(0,0,0), outline_w=2)

                    # 存檔節流
                    now = time.time()
                    if saving and (now - last_save_t >= 1.0/max(1.0, float(args.fps_save))):
                        fname = f"{subject}_{idx:06d}.jpg"
                        p = ALIGNED / fname
                        cv2.imwrite(str(p), aligned)
                        wr.writerow([
                            datetime.now().isoformat(timespec="seconds"),
                            str(p), subject, label, feats["phase"],
                            f"{Aw:.6f}", f"{Ca:.6f}", f"{Hp:.6f}",
                            f"{B:.6f}", f"{M:.6f}", f"{feats['magnitude']:.6f}", f"{feats['direction']:.6f}",
                            f"{feats['face_area']:.6f}", f"{feats['ipd']:.6f}",
                            f"{feats['ear_l']:.6f}", f"{feats['ear_r']:.6f}", f"{feats['mar']:.6f}",
                        ])
                        idx += 1
                        last_save_t = now
                        save_json(CALIB / f"{subject}.json", calib)

                cv2.imshow("aligned", aligned)

            # 原始畫面標示
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

        # 操作提示 & 錄製狀態
        if show_help:
            frame = draw_help_cn(frame)
        sflag = "● 正在錄製" if saving else "○ 待機"
        color = (0,0,255) if saving else (200,200,200)
        frame = put_ch_text(frame, sflag, (10, h-30), font=ZH_FONT, fill=color, outline=(0,0,0), outline_w=2)

        cv2.imshow("raw", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k in (ord('s'), ord('S')):
            saving = not saving
        elif k in (ord('h'), ord('H')):
            show_help = not show_help

    fcsv.close()
    cap.release(); cv2.destroyAllWindows()
    print("完成：", META)

if __name__ == "__main__":
    main()

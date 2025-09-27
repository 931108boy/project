# -*- coding: utf-8 -*-
"""
臉部資料蒐集（對齊 + 自動資料增強）
- Mediapipe FaceDetection 偵測 + 以雙眼做旋轉對齊
- 裁切到 224x224，按鍵存圖（1=理解、0=困惑）
- 每次存圖會同時輸出多個增強版本，擴充資料量
- ✅ 只允許 q / ESC 離開；按視窗叉叉不會退出（即使你把視窗關掉，會自動重建）
"""

import os, cv2, time, math, numpy as np
from datetime import datetime
import mediapipe as mp

# ============= 可調參數 =============
BASE_DIR = "yolo/datasets/train"   # 相對於你執行路徑
IMG_SIZE = 224
MARGIN = 0.35
AUG_ENABLED = True
AUG_PER_PRESS = 6
CAM_INDEX = 0
CLASS_MAP = {ord('1'): "understood", ord('0'): "confused"}
# ====================================

# 建資料夾
for cls in CLASS_MAP.values():
    os.makedirs(os.path.join(BASE_DIR, cls), exist_ok=True)

mp_fd = mp.solutions.face_detection
detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# （可選）中文顯示：Pillow
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
    CJK_FONTS = ["msjh.ttc", "msjhbd.ttc", "msyh.ttc", "mingliu.ttc", "NotoSansTC-Regular.otf"]
    def draw_text(img_bgr, text, xy, color=(255,255,255), size=22):
        font = None
        for name in CJK_FONTS:
            try:
                font = ImageFont.truetype(name, size); break
            except Exception: continue
        if font is None: font = ImageFont.load_default()
        b,g,r = color
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        ImageDraw.Draw(pil_img).text(xy, text, font=font, fill=(r,g,b))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
except Exception:
    PIL_OK = False
    def draw_text(img_bgr, text, xy, color=(255,255,255), size=22):
        cv2.putText(img_bgr, text.encode("ascii","ignore").decode("ascii"),
                    xy, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2); return img_bgr

def clamp(v, lo, hi): return max(lo, min(hi, v))

def safe_crop(img, x, y, w, h):
    H, W = img.shape[:2]
    x0 = clamp(int(x), 0, W-1); y0 = clamp(int(y), 0, H-1)
    x1 = clamp(int(x+w), 0, W);  y1 = clamp(int(y+h), 0, H)
    if x1 <= x0 or y1 <= y0: return None
    return img[y0:y1, x0:x1]

def get_eye_points(det):
    kps = det.location_data.relative_keypoints
    reye = kps[0]; leye = kps[1]
    return (reye.x, reye.y), (leye.x, leye.y)

def rotate_align_face(frame, bbox, reye_xy, leye_xy):
    H, W = frame.shape[:2]
    x, y, bw, bh = bbox
    rx, ry = reye_xy[0]*W, reye_xy[1]*H
    lx, ly = leye_xy[0]*W, leye_xy[1]*H
    dx, dy = (lx - rx), (ly - ry)
    angle = math.degrees(math.atan2(dy, dx))
    cx, cy = (rx + lx)/2.0, (ry + ly)/2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    corners = np.array([[x,y,1],[x+bw,y,1],[x,y+bh,1],[x+bw,y+bh,1]], dtype=np.float32)
    M3 = np.vstack([M,[0,0,1]]).astype(np.float32)
    rot = (M3 @ corners.T).T[:, :2]
    rx0, ry0 = rot[:,0].min(), rot[:,1].min()
    rx1, ry1 = rot[:,0].max(), rot[:,1].max()
    rw, rh = (rx1 - rx0), (ry1 - ry0)
    side = max(rw, rh) * (1.0 + 2*MARGIN)
    cx2, cy2 = (rx0 + rx1)/2.0, (ry0 + ry1)/2.0
    sy0 = cy2 - side/2 - side*0.10   # 上方多留
    sx0 = cx2 - side/2
    face = safe_crop(rotated, sx0, sy0, side, side)
    if face is None or face.size == 0: return None
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))

def random_augment(img):
    aug = img.copy()
    if np.random.rand() < 0.5: aug = cv2.flip(aug, 1)
    if np.random.rand() < 0.8:
        ang = np.random.uniform(-10,10)
        M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), ang, 1.0)
        aug = cv2.warpAffine(aug, M, (IMG_SIZE,IMG_SIZE),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() < 0.8:
        tx, ty = np.random.randint(-8,9), np.random.randint(-8,9)
        M = np.float32([[1,0,tx],[0,1,ty]])
        aug = cv2.warpAffine(aug, M, (IMG_SIZE,IMG_SIZE),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() < 0.9:
        alpha = np.random.uniform(0.9,1.1); beta = np.random.uniform(-15,15)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    if np.random.rand() < 0.5: aug = cv2.GaussianBlur(aug,(3,3),0)
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 6, aug.shape).astype(np.int16)
        aug = np.clip(aug.astype(np.int16)+noise, 0, 255).astype(np.uint8)
    return aug

def ensure_window(win):
    """如果視窗被關掉，重建它"""
    try:
        cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)

def main():
    win = "Affect Collector"
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): raise SystemExit("❌ 找不到攝影機")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    print("✅ 開始蒐集：1=理解  0=困惑  q / ESC=退出（按叉叉不會結束）")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)

            face_vis = None
            if res.detections:
                det = max(res.detections,
                          key=lambda d: d.location_data.relative_bounding_box.width
                                      * d.location_data.relative_bounding_box.height)
                box = det.location_data.relative_bounding_box
                x = int(box.xmin * W); y = int(box.ymin * H)
                bw = int(box.width * W); bh = int(box.height * H)

                mx, my = int(bw*MARGIN), int(bh*MARGIN*0.6)
                x2 = clamp(x - mx, 0, W-1); y2 = clamp(y - my, 0, H-1)
                bw2 = clamp(bw + 2*mx, 1, W - x2); bh2 = clamp(bh + 2*my, 1, H - y2)
                cv2.rectangle(frame, (x2,y2), (x2+bw2, y2+bh2), (0,255,0), 2)

                reye, leye = get_eye_points(det)
                face_vis = rotate_align_face(frame, (x2,y2,bw2,bh2), reye, leye)
                if face_vis is not None:
                    preview = cv2.resize(face_vis, (160,160))
                    frame[10:10+160, W-10-160:W-10] = preview

            msg = "1=理解  0=困惑  q=退出"
            frame = draw_text(frame, msg, (10, 30), (255,255,255), 24)
            if face_vis is None:
                frame = draw_text(frame, "未偵測到臉，請把臉移入畫面", (10, 60), (0,0,255), 22)

            # ------ 這裡改為 try/except，若視窗被關就自動重建 ------
            try:
                cv2.imshow(win, frame)
            except cv2.error:
                ensure_window(win)
                cv2.imshow(win, frame)
            # ---------------------------------------------------------

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q 或 ESC
                break

            if key in CLASS_MAP and face_vis is not None:
                cls = CLASS_MAP[key]
                out_dir = os.path.join(BASE_DIR, cls)
                os.makedirs(out_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                base_path = os.path.join(out_dir, f"{ts}.jpg")
                cv2.imwrite(base_path, face_vis)
                saved = 1
                if AUG_ENABLED and AUG_PER_PRESS > 0:
                    for i in range(AUG_PER_PRESS):
                        cv2.imwrite(os.path.join(out_dir, f"{ts}_aug{i+1}.jpg"),
                                    random_augment(face_vis))
                    saved += AUG_PER_PRESS
                print(f"💾 已存 {saved} 張 → {cls}（{base_path}）")

    finally:
        try: cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass

if __name__ == "__main__":
    main()

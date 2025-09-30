# yolo/realtime_infer.py
import os, cv2, math, numpy as np
from ultralytics import YOLO
import mediapipe as mp

WEIGHTS = "C:/Users/93110/OneDrive/桌面/project/專題/yolo/runs/classify/train_custom/weights/best.pt"  # 換成你的 best.pt 路徑
IMG_SIZE = 224
MARGIN = 0.35
CAM_INDEX = 0
LABEL_COLOR = (0, 255, 0)

mp_fd = mp.solutions.face_detection
detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)

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
    sy0 = cy2 - side/2 - side*0.10
    sx0 = cx2 - side/2

    face = safe_crop(rotated, sx0, sy0, side, side)
    if face is None or face.size == 0: return None
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))

def main():
    model = YOLO(WEIGHTS)   # 載入分類模型
    names = model.names     # 類別名稱字典，例如 {0:'confused', 1:'understood'}（依訓練而定）

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("❌ 找不到攝影機")

    win = "Affect Realtime"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    print("🎥 即時推論中：按 q / ESC 離開")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.process(rgb)

            label_text = "No face"
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
                    # Ultralytics classify：直接丟 numpy 圖片
                    r = model.predict(face_vis, imgsz=IMG_SIZE, verbose=False)[0]
                    # r.probs: tensor of class probs；r.names: dict
                    probs = r.probs.data.cpu().numpy()
                    cls_id = int(np.argmax(probs))
                    cls_name = r.names[cls_id]
                    p = float(probs[cls_id])
                    label_text = f"{cls_name}: {p*100:.1f}%"

                    # 預覽對齊臉
                    preview = cv2.resize(face_vis, (160,160))
                    frame[10:10+160, W-10-160:W-10] = preview

            # 畫面標示
            cv2.putText(frame, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, LABEL_COLOR, 2)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    finally:
        try: cap.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass

if __name__ == "__main__":
    main()

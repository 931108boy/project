# -*- coding: utf-8 -*-
"""
即時推論（中文 UI + 可用叉叉關閉 + 支援 --tag 載入模型）
優先順序（由高到低）：
1) --model_path / --scaler_path / --meta_path 三者同時提供 → 直接用
2) --tag <name> → 從 models/ 找出以 <name>_ 開頭、最新的一組檔案
3) 讀 models/latest.json

用法：
  python realtime_infer.py                         # 用 latest.json
  python realtime_infer.py --tag rf_v1            # 用 models/ 下 rf_v1_* 最新一組
  python realtime_infer.py --headless 1           # 無視窗模式
  python realtime_infer.py --model_path ... --scaler_path ... --meta_path ...
"""

import os, json, argparse, time, re
import numpy as np
import cv2
import mediapipe as mp
from joblib import load

# ---------- 中文繪字（Pillow） ----------
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False

CJK_FONTS = [
    "msjh.ttc", "msjhbd.ttc", "msyh.ttc", "mingliu.ttc", "simsun.ttc",
    "NotoSansTC-Regular.otf"
]

def draw_text_ch(img_bgr, text, xy, color=(255,255,255), size=24):
    if not PIL_OK:
        cv2.putText(img_bgr, text.encode("ascii","ignore").decode("ascii"),
                    xy, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img_bgr
    font = None
    for name in CJK_FONTS:
        try:
            font = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    b,g,r = color
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    ImageDraw.Draw(pil_img).text(xy, text, font=font, fill=(r,g,b))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------- 前處理（與訓練一致） ----------
def compute_face_size(landmarks, w, h):
    L = np.array([landmarks[234].x*w, landmarks[234].y*h], dtype=np.float32)
    R = np.array([landmarks[454].x*w, landmarks[454].y*h], dtype=np.float32)
    T = np.array([landmarks[10].x*w,  landmarks[10].y*h], dtype=np.float32)
    B = np.array([landmarks[152].x*w, landmarks[152].y*h], dtype=np.float32)
    return float(np.linalg.norm(L-R)), float(np.linalg.norm(T-B))

def landmarks_to_xyz_array(landmarks, w, h):
    N = len(landmarks)
    out = np.zeros((N,3), dtype=np.float32)
    for i,p in enumerate(landmarks):
        out[i,0] = p.x * w
        out[i,1] = p.y * h
        out[i,2] = p.z * w  # 讓 z 與 x 同量級
    return out

def normalized_delta(curr_xyz, base_xyz, face_w, face_h):
    delta = curr_xyz - base_xyz
    N = delta.shape[0]
    denom = np.stack([
        np.full(N, max(face_w,1e-6), dtype=np.float32),
        np.full(N, max(face_h,1e-6), dtype=np.float32),
        np.full(N, max(face_w,1e-6), dtype=np.float32)
    ], axis=1)
    return delta / denom

# ---------- 載入模型 ----------
TS_PATTERN = re.compile(r".*_(\d{8}_\d{6})_(?:model|scaler|feat_meta)\.(?:joblib|json)$")

def find_latest_by_tag(tag: str, models_dir="models"):
    """
    找出 models/ 下以 '<tag>_' 開頭且時間戳最新的「成套」檔案：
    <tag>_<model>_<YYYYMMDD_HHMMSS>_model.joblib
    <tag>_<model>_<YYYYMMDD_HHMMSS>_scaler.joblib
    <tag>_<model>_<YYYYMMDD_HHMMSS>_feat_meta.json
    回傳 (scaler_path, model_path, meta_path)
    """
    if not os.path.isdir(models_dir):
        raise SystemExit(f"❌ 找不到資料夾：{models_dir}")

    candidates = []
    for fname in os.listdir(models_dir):
        if not fname.startswith(f"{tag}_"):
            continue
        if fname.endswith("_model.joblib"):
            stem = fname.replace("_model.joblib", "")
            # 推導另外兩個路徑
            model_path  = os.path.join(models_dir, f"{stem}_model.joblib")
            scaler_path = os.path.join(models_dir, f"{stem}_scaler.joblib")
            meta_path   = os.path.join(models_dir, f"{stem}_feat_meta.json")
            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path):
                # 取時間戳排序（若沒匹配到就用 mtime 當備援）
                m = TS_PATTERN.match(fname)
                ts_key = m.group(1) if m else str(os.path.getmtime(model_path))
                candidates.append((ts_key, scaler_path, model_path, meta_path))

    if not candidates:
        raise SystemExit(f"❌ models/ 下找不到符合 tag='{tag}' 的完整模型三件套")

    # 依時間戳降序
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, scaler_path, model_path, meta_path = candidates[0]
    return scaler_path, model_path, meta_path

def load_artifacts(args):
    # 1) 明確指定三個路徑
    if args.model_path and args.scaler_path and args.meta_path:
        model_path, scaler_path, meta_path = args.model_path, args.scaler_path, args.meta_path
    # 2) 用 tag 尋找最新組
    elif args.tag:
        scaler_path, model_path, meta_path = find_latest_by_tag(args.tag, models_dir="models")
    # 3) 退回 latest.json
    else:
        latest = os.path.join("models", "latest.json")
        if not os.path.exists(latest):
            raise SystemExit("❌ 找不到 models/latest.json，且未指定 --tag 或三個路徑。")
        with open(latest, "r", encoding="utf-8") as f:
            j = json.load(f)
        scaler_path, model_path, meta_path = j["scaler"], j["model"], j["meta"]

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
        raise SystemExit("❌ 模型檔路徑不完整，請確認 model/scaler/meta 三件檔案都存在。")

    model = load(model_path)
    scaler = load(scaler_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_cols = meta["feature_names"]
    print("✅ 已載入：")
    print("  scaler:", scaler_path)
    print("  model :", model_path)
    print("  meta  :", meta_path)
    return scaler, model, feat_cols

# ---------- 參數 ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",  type=str, default="", help="模型 .joblib 路徑")
    ap.add_argument("--scaler_path", type=str, default="", help="標準化器 .joblib 路徑")
    ap.add_argument("--meta_path",   type=str, default="", help="特徵欄位 .json 路徑")
    ap.add_argument("--tag",         type=str, default="", help="模型檔名前綴標籤（例如 rf_v1）")
    ap.add_argument("--headless",    type=int, default=0,  help="1=無視窗模式")
    ap.add_argument("--cam",         type=int, default=0,  help="攝影機 index")
    ap.add_argument("--interval",    type=float, default=0.5, help="終端輸出間隔秒數")
    return ap.parse_args()

def main():
    args = parse_args()
    scaler, model, feat_cols = load_artifacts(args)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("❌ 找不到攝影機")

    WIN = "Affect Realtime"
    if not args.headless:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 960, 540)

    baseline_xyz = None
    last_print = 0.0

    print("操作：b=校準（3秒中性臉）  q=離開（視窗叉叉也可）")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not args.headless:
            frame = draw_text_ch(frame, "b=校準（3秒中性臉）  q=離開", (10, 28), (255,255,0), 26)
            if baseline_xyz is None:
                frame = draw_text_ch(frame, "⚠ 尚未校準，請按 b 並保持中性表情", (10, 58), (0,0,255), 22)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            if baseline_xyz is not None:
                curr_xyz = landmarks_to_xyz_array(lm, w, h)
                face_w, face_h = compute_face_size(lm, w, h)
                nd = normalized_delta(curr_xyz, baseline_xyz, face_w, face_h)
                flat = nd.reshape(-1)

                feat = {"face_width": face_w, "face_height": face_h}
                N = nd.shape[0]
                for i in range(N):
                    feat[f"dx_{i}"] = flat[i*3 + 0]
                    feat[f"dy_{i}"] = flat[i*3 + 1]
                    feat[f"dz_{i}"] = flat[i*3 + 2]
                x = np.array([feat.get(c, 0.0) for c in feat_cols], dtype=np.float32).reshape(1, -1)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                xs = scaler.transform(x)

                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(xs)[0, 1])
                else:
                    prob = float(model.decision_function(xs).ravel()[0]) if hasattr(model, "decision_function") else 0.0
                pred = int(model.predict(xs)[0])
                label = "理解" if pred == 1 else "困惑"

                now = time.time()
                if now - last_print >= args.interval:
                    print(f"[{time.strftime('%H:%M:%S')}] {label}  機率={prob:.3f}")
                    last_print = now

                if not args.headless:
                    color = (0,255,0) if pred==1 else (0,0,255)
                    frame = draw_text_ch(frame, f"{label}  機率={prob:.2f}", (10, 92), color, 28)

        if not args.headless:
            cv2.imshow(WIN, frame)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(1) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        if key == ord('b'):
            print("➡ 校準 3 秒：請保持中性表情…")
            end_t = time.time() + 3.0
            buf = []
            while time.time() < end_t:
                ok2, frm = cap.read()
                if not ok2: break
                rs = face_mesh.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                if rs.multi_face_landmarks:
                    lm2 = rs.multi_face_landmarks[0].landmark
                    buf.append(landmarks_to_xyz_array(lm2, frm.shape[1], frm.shape[0]))
                    if not args.headless:
                        frm = draw_text_ch(frm, "校準中…請保持中性表情", (10, 28), (255,255,0), 26)
                else:
                    if not args.headless:
                        frm = draw_text_ch(frm, "未偵測到臉，請保持臉在畫面中", (10, 28), (0,0,255), 26)
                if not args.headless:
                    cv2.imshow(WIN, frm); cv2.waitKey(1)
            if len(buf) >= 5:
                baseline_xyz = np.mean(np.stack(buf, axis=0), axis=0)
                print("✅ 校準完成！")
            else:
                print("❌ 校準失敗，請重試")

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

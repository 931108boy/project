# -*- coding: utf-8 -*-
"""
單次點名（先預覽、按鍵拍照再比對）
- 直接執行，不需參數
- 操作：Space=拍照、C=3秒倒數自動拍、R=重拍、Q=離開
- 支援中文路徑（用臨時 ASCII 陰影庫）
- 偵測器多重 fallback：retinaface→mediapipe→opencv→relaxed
"""

import os, cv2, json, shutil, uuid, tempfile, time
from datetime import datetime
from pathlib import Path
import numpy as np
from deepface import DeepFace

# ============== 可調參數 ==============
MODEL_NAME   = "SFace"
DIST_METRIC  = "cosine"
THRESH       = 0.40
CAM_INDEX    = 0
SAVE_CAPTURE = False             # 是否把拍到的那張存到 captures/
DETECTORS_TRY = ["retinaface", "mediapipe", "opencv"]
# =====================================

HERE        = Path(__file__).resolve().parent
DEFAULT_DB  = str(HERE / "faces_db")
CAPTURE_DIR = str(HERE / "captures")


# -------- Unicode 安全 I/O（支援中文檔名） --------
def imread_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_unicode(path: str, img, quality=95) -> bool:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(path)
        return True
    except Exception:
        return False


# ----------------- DB 檢查 / 建庫 ------------------
def db_is_ready(db_dir: str) -> bool:
    if not os.path.isdir(db_dir):
        return False
    for name in os.listdir(db_dir):
        sub = os.path.join(db_dir, name)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    return True
    return False

def bootstrap_build_db(db_dir: str, cam_index: int):
    os.makedirs(db_dir, exist_ok=True)
    person = input("資料庫為空，請輸入要新增的人名/學號：").strip()
    if not person:
        raise SystemExit("❌ 未輸入人名，無法建立資料庫")

    out_dir = os.path.join(db_dir, person)
    os.makedirs(out_dir, exist_ok=True)

    print("➡ 建庫模式：請擺好臉，按【空白鍵】存圖，按【q】結束（多角度、不同表情各存幾張）")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("❌ 找不到攝影機")

    win = f"Build DB - {person}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.putText(frame, f"{person} | Space=Save   Q=Finish",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            if k == 32:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(out_dir, f"{ts}.jpg")
                if imwrite_unicode(path, frame):
                    saved += 1
                    print(f"💾 已存第 {saved} 張：{os.path.abspath(path)}")
                else:
                    print("❌ 寫檔失敗（檢查權限/路徑）")
    finally:
        try: cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass

    if saved == 0:
        raise SystemExit("❌ 沒有存到任何圖，無法建立資料庫")
    print(f"✅ 建庫完成：{out_dir}（共 {saved} 張）")


# ---- 建立「臨時 ASCII 陰影 DB」並建立映射 ----
def build_ascii_shadow_db(db_dir: str):
    tmp_root = Path(tempfile.gettempdir()) / f"tmp_db_ascii_{uuid.uuid4().hex[:8]}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    mapping = {}
    person_idx = 1

    for name in os.listdir(db_dir):
        src_person = os.path.join(db_dir, name)
        if not os.path.isdir(src_person): continue
        slug = "".join(ch if ord(ch) < 128 else "_" for ch in name).strip().replace(" ", "_")
        slug = f"p{person_idx:03d}_{slug or 'person'}"
        person_idx += 1

        dst_person = tmp_root / slug
        dst_person.mkdir(parents=True, exist_ok=True)
        mapping[dst_person.name] = name

        for f in os.listdir(src_person):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
            img = imread_unicode(os.path.join(src_person, f))
            if img is None: continue
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            imwrite_unicode(str(dst_person / f"{ts}.jpg"), img)

    return str(tmp_root), mapping, str(tmp_root)


# ----------------- 視覺化輔助 -----------------
def draw_hud(img, text_lines, color=(50,255,50)):
    y = 30
    for t in text_lines:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        y += 34
    return img

def countdown_overlay(frame, seconds=3):
    show = frame.copy()
    h, w = show.shape[:2]
    for sec in range(seconds, 0, -1):
        img = show.copy()
        cv2.putText(img, str(sec), (w//2-20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,0,255), 5)
        cv2.imshow("Capture", img)
        cv2.waitKey(1000)


# -------------------- 主流程 --------------------
def main():
    db_path = DEFAULT_DB
    print("CWD：", os.getcwd())
    print("使用的資料庫 DB：", os.path.abspath(db_path))

    if not db_is_ready(db_path):
        bootstrap_build_db(db_path, CAM_INDEX)

    # 建臨時 ASCII 陰影庫
    shadow_db, name_map, temp_root = build_ascii_shadow_db(db_path)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        _cleanup_tmp(temp_root)
        raise SystemExit("❌ 找不到攝影機")

    cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture", 960, 540)

    captured = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            hud = [
                "準備拍照：Space=拍照   C=倒數3秒   R=重拍   Q=離開",
                f"門檻(threshold)={THRESH:.2f}  模型={MODEL_NAME}  度量={DIST_METRIC}"
            ]
            show = draw_hud(frame.copy(), hud, (255,255,255))
            cv2.imshow("Capture", show)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                _cleanup_tmp(temp_root)
                return
            if k == ord('c'):
                countdown_overlay(frame, 3)
                captured = frame.copy()
            if k == 32:  # Space
                captured = frame.copy()

            if captured is not None:
                break

        # 顯示「拍到了」並可選擇重拍
        while True:
            preview = captured.copy()
            cv2.putText(preview, "已拍到：R=重拍  Enter=確認", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            cv2.imshow("Capture", preview)
            k = cv2.waitKey(1) & 0xFF
            if k in (13, 10):  # Enter
                break
            if k == ord('r'):
                captured = None
                # 回到取景
                return main()
            if k in (ord('q'), 27):
                _cleanup_tmp(temp_root); return

    finally:
        try: cap.release()
        except Exception: pass
        # 不先關窗，後面還要顯示結果

    # 需要的話存檔
    image_path = ""
    if SAVE_CAPTURE:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = str(Path(CAPTURE_DIR) / f"capture_{ts}.jpg")
        imwrite_unicode(image_path, captured)

    # DeepFace 查詢（多偵測器 fallback）
    dfs = None
    for det in DETECTORS_TRY + ["relaxed"]:
        try:
            if det == "relaxed":
                dfs = DeepFace.find(
                    img_path=captured,
                    db_path=shadow_db,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTORS_TRY[0],
                    distance_metric=DIST_METRIC,
                    enforce_detection=False,
                    silent=True
                )
            else:
                dfs = DeepFace.find(
                    img_path=captured,
                    db_path=shadow_db,
                    model_name=MODEL_NAME,
                    detector_backend=det,
                    distance_metric=DIST_METRIC,
                    enforce_detection=True,
                    silent=True
                )
            if isinstance(dfs, list) and len(dfs) > 0:
                break
        except Exception:
            dfs = None

    # 解析結果
    name, dist, status = "", None, "noface"
    try:
        if isinstance(dfs, list) and len(dfs) > 0 and len(dfs[0]) > 0:
            row0 = dfs[0].iloc[0]
            dist = float(row0["distance"])
            ascii_person = os.path.basename(os.path.dirname(row0["identity"]))
            orig_name = name_map.get(ascii_person, ascii_person)
            if dist <= THRESH:
                name, status = orig_name, "matched"
            else:
                name, status = "Unknown", "unknown"
        else:
            status = "noface"
    except Exception:
        status = "noface"

    result = {
        "status": status,
        "name": name,
        "distance": dist,
        "threshold": THRESH,
        "image_path": image_path,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    print(json.dumps(result, ensure_ascii=False))

    # 顯示結果
    try:
        res_img = captured.copy()
        if status == "matched":
            line1 = "Status: matched"; color=(0,200,0)
            line2 = f"Name: {name}"
            line3 = f"Distance: {dist:.3f}  (<= {THRESH:.2f} = match)"
        elif status == "unknown":
            line1 = "Status: unknown"; color=(0,165,255)
            line2 = "Name: Unknown"
            line3 = f"Distance: {dist:.3f}  (> {THRESH:.2f} = not match)"
        else:
            line1 = "Status: noface"; color=(0,0,255)
            line2 = "No face detected"; line3 = ""

        y=30
        for txt in [line1, line2, line3]:
            if not txt: continue
            cv2.putText(res_img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y += 34
        cv2.imshow("Result - press any key", res_img)
        cv2.waitKey(0)
    finally:
        try: cv2.destroyAllWindows()
        except Exception: pass

    _cleanup_tmp(temp_root)


def _cleanup_tmp(temp_root: str):
    try:
        shutil.rmtree(temp_root, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
簡單截圖工具：建立 faces_db（固定與程式同層）
- n：輸入 / 切換人名
- s 或 空白鍵：存圖（若未設定人名會先詢問）
- q 或 ESC：離開（視窗叉叉也可）
- 使用 imencode().tofile() 以支援含中文的路徑（解決 imwrite 失敗）
"""

import os
import cv2
import time
import numpy as np
from datetime import datetime

# 固定輸出資料夾（與程式同一層）
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces_db")
os.makedirs(BASE_DIR, exist_ok=True)

def ensure_person_dir(person: str) -> str:
    out_dir = os.path.join(BASE_DIR, person)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def safe_write_jpg(path: str, image) -> bool:
    """
    Windows 上 OpenCV 的 imwrite 對中文路徑常失敗；
    用 imencode('.jpg', img)[1].tofile(path) 可安全寫入 Unicode 路徑。
    """
    try:
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return False
        # 用 numpy 的 tofile 可處理 Unicode 路徑
        buf.tofile(path)
        return True
    except Exception:
        return False

def overlay_text(img, text, pos=(10, 60), color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    print("📁 圖片會儲存在：", os.path.abspath(BASE_DIR))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("❌ 找不到攝影機（請檢查相機權限與索引）")

    win = "Face DB Collector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    current_name = None
    last_saved_msg = ""
    last_saved_time = 0.0

    print("⚡ 操作：n=輸入名字 / s 或 [空白鍵]=存圖 / q 或 ESC=退出（視窗叉叉也可）")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            status = f"當前人物: {current_name if current_name else '未設定 (請按 n)'} | s/Space=存圖, n=改名, q/ESC=退出"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if last_saved_msg and (time.time() - last_saved_time) < 0.7:
                overlay_text(frame, last_saved_msg, (10, 60), (0, 255, 0))

            cv2.imshow(win, frame)

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break

            if k == ord('n'):
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                name = input("請輸入人名或學號：").strip()
                if name:
                    current_name = name
                    out_dir = ensure_person_dir(current_name)
                    cv2.setWindowTitle(win, f"Face DB Collector - {current_name}")
                    print(f"📂 已切換至：{os.path.abspath(out_dir)}")
                else:
                    print("（未變更）")

            if k == ord('s') or k == 32:
                if not current_name:
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    name = input("尚未設定人物。請輸入人名/學號：").strip()
                    if not name:
                        print("⚠️ 未輸入人名，取消存圖")
                        continue
                    current_name = name
                    out_dir = ensure_person_dir(current_name)
                    cv2.setWindowTitle(win, f"Face DB Collector - {current_name}")
                    print(f"📂 已切換至：{os.path.abspath(out_dir)}")
                else:
                    out_dir = ensure_person_dir(current_name)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(out_dir, f"{ts}.jpg")

                ok_write = safe_write_jpg(path, frame)
                if not ok_write:
                    print(f"❌ 寫檔失敗：{path}（請檢查磁碟空間/權限/路徑）")
                    last_saved_msg = "❌ 寫檔失敗"
                else:
                    print(f"💾 已存：{os.path.abspath(path)}")
                    last_saved_msg = f"已存：{os.path.basename(path)}"
                last_saved_time = time.time()

    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()

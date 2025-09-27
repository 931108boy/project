# -*- coding: utf-8 -*-
"""
收集臉部表情資料（理解=1 / 困惑=0）
------------------------------------------------
核心想法：
  1) 先用 3 秒「中性臉」當 Baseline（基準），計算每個 landmark 的平均 (x,y,z)。
  2) 之後每次標註時，取「當前臉 - Baseline」得到 Δx,Δy,Δz，
     再用「臉寬 / 臉高」做比例正規化，降低不同人臉大小與相機距離的差異。
  3) 寫入 CSV：time, label, face_width, face_height, dx_0,dy_0,dz_0, ... , dx_(N-1),dy_(N-1),dz_(N-1)

操作：
  b = 校準（請維持 3 秒中性表情）
  1 = 標註「理解」
  0 = 標註「困惑」
  s = 略過不存
  q 或按視窗叉叉 = 離開

輸出：
  <程式同資料夾>/data/all_features.csv
  ※ CSV 標頭會依「實際 landmark 點數 N（可能 468 或 478）」動態建立
"""

# ----------------------------
# 1) 匯入標準函式庫與第三方套件
# ----------------------------
import os               # 處理路徑 / 檔案
import time             # 取得 UNIX 時間戳（每筆樣本的時間）
import csv              # 寫 CSV 檔
import numpy as np      # 數值運算（平均、向量距離、陣列處理）
import cv2              # OpenCV（攝影機、畫圖、視窗）
import mediapipe as mp  # MediaPipe（FaceMesh 臉部關鍵點）

# ----------------------------
# 2) 中文繪字設定（避免 OpenCV 中文亂碼成「？？？」）
# ----------------------------
try:
    # Pillow 可在影像上畫任意字型
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    # 若沒裝 Pillow，會退回 OpenCV 英文字（中文會被忽略）
    PIL_OK = False

# 常見中文字型清單（會依序嘗試載入到找到為止；
# 你也可以把字型檔丟到程式資料夾，並把檔名加進來）
CJK_FONTS = [
    "msjh.ttc",              # 微軟正黑體
    "msjhbd.ttc",            # 微軟正黑體-粗
    "msyh.ttc",              # 微軟雅黑
    "mingliu.ttc",           # 新細明體
    "simsun.ttc",            # 中易宋體
    "NotoSansTC-Regular.otf" # Google Noto Sans TC
]

def draw_text_ch(img_bgr, text, xy, color=(255, 255, 255), size=24):
    """
    在 OpenCV 的 BGR 影像上畫『中文文字』。
    參數：
      img_bgr: BGR 影像（cv2）
      text:    要顯示的字串（可含中文）
      xy:      文字左上角 (x, y) 座標
      color:   BGR 顏色（例如白色 (255,255,255)）
      size:    字體大小
    """
    if not PIL_OK:
        # 沒有 Pillow 時，以英文 fallback（中文會被去掉避免亂碼）
        cv2.putText(
            img_bgr,
            text.encode("ascii", "ignore").decode("ascii"),
            xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
        return img_bgr

    # 嘗試載入任何一套中文字型
    font = None
    for name in CJK_FONTS:
        try:
            font = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if font is None:
        # 找不到中文字型就退回系統預設（可能仍無法顯示中文，但不會當）
        font = ImageFont.load_default()

    # Pillow 用 RGB；OpenCV 用 BGR → 要轉色兩次
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    # Pillow 的 fill 是 RGB，這裡把 BGR → RGB
    b, g, r = color
    draw.text(xy, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ----------------------------
# 3) 檔案與資料夾設定（固定寫在程式旁的 data/ 下）
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 取得程式所在資料夾的絕對路徑
DATA_DIR = os.path.join(BASE_DIR, "data")              # data 子資料夾
os.makedirs(DATA_DIR, exist_ok=True)                   # 若不存在就建立
CSV_PATH = os.path.join(DATA_DIR, "all_features.csv")  # 單一整合檔（持續追加）

# CSV 標頭是否已建立（如果檔案已存在，視為已建立；但仍需等第一次偵測到臉以取得 N）
csv_header_written = os.path.exists(CSV_PATH)

# ----------------------------
# 4) MediaPipe FaceMesh 初始化（solutions API）
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh  # 模組入口
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,   # 串流模式（攝影機）
    max_num_faces=1,           # 只追蹤一張臉（收資料較穩定）
    refine_landmarks=True,     # 使用精細模型（眼睛/唇更準確；點數可能 478）
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# 5) Baseline（中性臉）與點數 N（468/478）變數
# ----------------------------
baseline_xyz = None  # 形狀 (N,3) 的 numpy 陣列：baseline 的 (x,y,z)（單位：像素）
BASE_N = None        # landmark 點數（第一次偵測臉時決定：468 或 478）

def compute_face_size(landmarks, w, h):
    """
    計算臉的寬度與高度（單位：像素）。
    用固定的四個 landmark：
      - 臉寬：234（左臉頰）到 454（右臉頰）
      - 臉高：10（額頭）到 152（下巴）
    """
    L = np.array([landmarks[234].x * w, landmarks[234].y * h], dtype=np.float32)
    R = np.array([landmarks[454].x * w, landmarks[454].y * h], dtype=np.float32)
    T = np.array([landmarks[10].x  * w, landmarks[10].y  * h], dtype=np.float32)
    B = np.array([landmarks[152].x * w, landmarks[152].y * h], dtype=np.float32)
    face_w = float(np.linalg.norm(L - R))  # 左右臉頰的歐式距離 → 臉寬
    face_h = float(np.linalg.norm(T - B))  # 額頭到下巴的距離     → 臉高
    return face_w, face_h

def landmarks_to_xyz_array(landmarks, w, h):
    """
    將 MediaPipe 的 landmarks 轉為 (N,3) numpy 陣列（單位一律轉成像素）：
      x = p.x * w
      y = p.y * h
      z = p.z * w   ← 用畫面寬 w 當尺度，讓 z 與 x 量級一致，後續較好做比例化
    """
    N = len(landmarks)                                # 動態取得點數（468 / 478）
    out = np.zeros((N, 3), dtype=np.float32)          # 建立 (N,3) 陣列
    for i, p in enumerate(landmarks):
        out[i, 0] = p.x * w                           # x 轉像素
        out[i, 1] = p.y * h                           # y 轉像素
        out[i, 2] = p.z * w                           # z 也轉成像素量級
    return out

def normalized_delta(curr_xyz, base_xyz, face_w, face_h):
    """
    計算『相對差異 + 比例正規化』特徵，形狀仍為 (N,3)：
      Δ = curr_xyz - base_xyz                      # 逐點差異（像素）
      dx = Δx / face_w                             # 水平差異用臉寬正規化
      dy = Δy / face_h                             # 垂直差異用臉高正規化
      dz = Δz / face_w                             # 深度差異也用臉寬正規化
    這樣可減輕個人臉型大小 / 鏡頭距離的影響。
    """
    delta = curr_xyz - base_xyz                     # 逐點差異
    N = delta.shape[0]
    # 建立 (N,3) 的分母陣列，避免分母為 0
    denom = np.stack([
        np.full(N, max(face_w, 1e-6), dtype=np.float32),  # x 用臉寬
        np.full(N, max(face_h, 1e-6), dtype=np.float32),  # y 用臉高
        np.full(N, max(face_w, 1e-6), dtype=np.float32)   # z 用臉寬
    ], axis=1)
    return delta / denom

# ----------------------------
# 6) 建立 OpenCV 視窗（避免中文亂碼 → 視窗標題用英文；中文 UI 畫在畫面上）
# ----------------------------
WIN = "Affect Collector"              # 英文視窗名最穩
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)  # 可調整大小
cv2.resizeWindow(WIN, 1280, 800)         # 放大以避免底部字被裁切

# ----------------------------
# 7) 開啟攝影機
# ----------------------------
cap = cv2.VideoCapture(0)     # 打開預設攝影機（index=0）
if not cap.isOpened():
    raise SystemExit("❌ 找不到攝影機（請檢查裝置或權限）")

print("操作：b=校準  1=理解  0=困惑  s=略過  q=退出（視窗叉叉也可）")

# ----------------------------
# 8) 主迴圈：逐幀處理
# ----------------------------
while True:
    ok, frame = cap.read()           # 讀取一幀影像
    if not ok:
        break
    h, w = frame.shape[:2]           # 當前畫面高度與寬度（像素）

    # MediaPipe 的輸入要求是 RGB；OpenCV 讀到的是 BGR → 需轉色
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FaceMesh 推論：取得臉部 landmarks（若沒偵測到臉，res.multi_face_landmarks 會是 None）
    res = face_mesh.process(rgb)

    # 在畫面上畫中文 UI（用 Pillow，避免問號）
    frame = draw_text_ch(frame, "表情資料蒐集（相對差異 + 比例正規化）", (10, 28), (255, 255, 0), 28)
    frame = draw_text_ch(frame, "b=校準  1=理解  0=困惑  s=略過  q=退出", (10, 62), (200, 255, 200), 24)
    if baseline_xyz is None:
        # 若還沒校準 Baseline，提醒使用者先按 b
        frame = draw_text_ch(frame, "⚠ 請先按 b，保持中性表情 3 秒完成校準", (10, 96), (0, 0, 255), 24)

    # 若偵測到臉：畫一些提示點（讓你眼見為憑）
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark   # 取得第一張臉的 landmarks
        # 畫幾個常用點：左眼(上/下)、右眼(上/下)、嘴角(左/右)
        for idx in [159, 145, 386, 374, 61, 291]:
            px = int(lm[idx].x * w)
            py = int(lm[idx].y * h)
            cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        # 第一次看到臉時，我們就知道 N（點數）。若 CSV 還沒寫標頭 → 現在建立。
        if (BASE_N is None):
            BASE_N = len(lm)  # 實際 landmark 點數（468 或 478）
            # 如果 CSV 檔還沒建立過標頭，就依 N 動態寫入欄位名
            if not csv_header_written:
                with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                    wcsv = csv.writer(f)
                    header = ["time", "label", "face_width", "face_height"]
                    for i in range(BASE_N):
                        header += [f"dx_{i}", f"dy_{i}", f"dz_{i}"]
                    wcsv.writerow(header)
                csv_header_written = True
            # 畫出提示：目前偵測到幾個點
            frame = draw_text_ch(frame, f"偵測到 {BASE_N} 個關鍵點", (10, 130), (180, 220, 255), 22)

    # 顯示畫面
    cv2.imshow(WIN, frame)

    # 允許用「右上角叉叉」關閉視窗：
    # 若視窗被關閉（可視屬性 < 1），直接跳出迴圈
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

    # 讀鍵盤：每回圈等待 1ms；回傳按鍵 ASCII（& 0xFF 避免平台差異）
    key = cv2.waitKey(1) & 0xFF

    # q 或 ESC 離開
    if key == ord('q') or key == 27:
        break

    # ------------- 校準（b）：收 3 秒中性表情做平均 -------------
    if key == ord('b'):
        print("➡ 開始校準：請保持中性表情 3 秒…")
        t_end = time.time() + 3.0      # 校準時長（秒）
        buf_xyz = []                   # 每幀的 (N,3) 會放在這
        while time.time() < t_end:
            ok2, frm = cap.read()
            if not ok2:
                break
            h2, w2 = frm.shape[:2]
            rs = face_mesh.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            if rs.multi_face_landmarks:
                lm2 = rs.multi_face_landmarks[0].landmark
                # 如果還不知道 N（第一次校準時可能發生），此時設定 BASE_N
                if BASE_N is None:
                    BASE_N = len(lm2)
                    # 若 CSV 還沒寫標頭，也在此刻建立
                    if not csv_header_written:
                        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                            wcsv = csv.writer(f)
                            header = ["time", "label", "face_width", "face_height"]
                            for i in range(BASE_N):
                                header += [f"dx_{i}", f"dy_{i}", f"dz_{i}"]
                            wcsv.writerow(header)
                        csv_header_written = True
                # 轉為像素座標陣列 (N,3)
                arr = landmarks_to_xyz_array(lm2, w2, h2)
                buf_xyz.append(arr)
                # 在畫面標示「校準中」
                frm = draw_text_ch(frm, "校準中…請保持中性表情", (10, 60), (255, 255, 0), 26)
            else:
                # 若臉暫時偵測不到，提醒把臉對準鏡頭
                frm = draw_text_ch(frm, "未偵測到臉，請保持臉在畫面中", (10, 60), (0, 0, 255), 26)
            cv2.imshow(WIN, frm)
            cv2.waitKey(1)  # 讓畫面能更新

        # 校準結束：如果收集到足夠幀，取平均當 baseline
        if len(buf_xyz) >= 5:  # 至少 5 幀才算成功
            baseline_xyz = np.mean(np.stack(buf_xyz, axis=0), axis=0)  # (N,3)
            print(f"✅ 校準完成！基準點數 N = {baseline_xyz.shape[0]}")
        else:
            print("❌ 校準失敗（臉偵測不足）。請重按 b 重試。")

    # ------------- 標註寫檔：1=理解, 0=困惑, s=略過 -------------
    if key in (ord('1'), ord('0'), ord('s')):
        # 必須：已校準過 baseline 且此幀有偵測到臉
        if (baseline_xyz is not None) and res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            # 1) 取得此幀的 (N,3) 像素座標
            curr_xyz = landmarks_to_xyz_array(lm, w, h)
            # 2) 計算此幀的臉寬 / 臉高（像素，記錄下來供後續分析）
            face_w, face_h = compute_face_size(lm, w, h)
            # 3) 計算「相對差異 + 比例正規化」(N,3)
            nd = normalized_delta(curr_xyz, baseline_xyz, face_w, face_h)

            if key != ord('s'):  # 's' 表示略過，不寫檔
                # 4) 決定標籤：理解(1) 或 困惑(0)
                label = 1 if key == ord('1') else 0

                # 5) 將 (N,3) 攤平成一維：[dx0,dy0,dz0, dx1,dy1,dz1, ...]
                flat = nd.reshape(-1).tolist()

                # 6) 追加寫入 CSV：time, label, face_w, face_h, flat...
                with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                    wcsv = csv.writer(f)
                    row = [time.time(), label, face_w, face_h] + flat
                    wcsv.writerow(row)

                print("📌 已存一筆：", "理解(1)" if label == 1 else "困惑(0)",
                      f"；點數 N={nd.shape[0]} → 維度={len(flat)}")
            else:
                print("⏭ 略過此幀（不寫入）")
        else:
            print("⚠ 尚未校準或此幀未偵測到臉，無法寫入。請先按 b 校準，並確保臉在畫面中。")

# ----------------------------
# 9) 收尾：釋放資源
# ----------------------------
cap.release()              # 關閉攝影機
cv2.destroyAllWindows()    # 關閉所有 OpenCV 視窗
print("👋 結束。資料已寫入：", CSV_PATH)

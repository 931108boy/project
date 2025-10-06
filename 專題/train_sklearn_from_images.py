# -*- coding: utf-8 -*-
"""
把原本 YOLO 用的影像資料夾（class 子資料夾）轉為 scikit-learn 可用的特徵，
並訓練/評估一個分類器（LogisticRegression）。
支援兩種特徵：
  1) landmarks：用 MediaPipe FaceMesh 萃取臉部幾何特徵（推薦）
  2) pixels   ：灰階+縮放後攤平成向量（快速基線）

資料夾格式：
data/
  understood/*.jpg|png|jpeg
  confused/*.jpg|png|jpeg
"""

import argparse, sys, os, math
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

# --------- MediaPipe (僅在 method=landmarks 需要) ----------
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images_by_class(data_dir: Path):
    """走訪 data_dir 下面的每個子資料夾，子資料夾名視為 label。"""
    for cls_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        label = cls_dir.name
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() in IMG_EXTS:
                yield img_path, label


# =========================
# 方法一：像素攤平特徵
# =========================
def extract_pixels_feature(img_bgr, img_size=64):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    feat = (gray.astype(np.float32) / 255.0).flatten()
    return feat


# =========================
# 方法二：FaceMesh 幾何特徵
# =========================
def _dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def _safe_ratio(a, b, eps=1e-6):
    return float(a / (b + eps))

def extract_landmarks_feature(img_bgr, face_mesh):
    """回傳一組描述臉部幾何的特徵；偵測失敗回傳 None。"""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    # 取第一張臉
    lm = res.multi_face_landmarks[0].landmark
    # 把需要的 2D 座標轉成像素座標（x,y）
    def pt(i):
        return (lm[i].x * w, lm[i].y * h)

    # 參考點（MediaPipe FaceMesh 索引）
    # 眼角：左眼外側 33、右眼外側 263；左眼內側 133、右眼內側 362
    # 眼睛其他點（計算 EAR 常用）：左 160,158,144,153；右 387,385,373,380
    L_OUT, L_IN = 33, 133
    R_OUT, R_IN = 263, 362
    L_UP1, L_UP2, L_LO1, L_LO2 = 160, 158, 144, 153
    R_UP1, R_UP2, R_LO1, R_LO2 = 387, 385, 373, 380

    # 嘴角：左 61、右 291；上唇 13、下唇 14（近似 MAR）
    M_LEFT, M_RIGHT, M_UP, M_LOW = 61, 291, 13, 14

    try:
        # EAR（eye aspect ratio），越小越像閉眼
        def ear(left=True):
            if left:
                p1, p2 = pt(L_OUT), pt(L_IN)
                pv1, pv2, pv3, pv4 = pt(L_UP1), pt(L_UP2), pt(L_LO1), pt(L_LO2)
            else:
                p1, p2 = pt(R_OUT), pt(R_IN)
                pv1, pv2, pv3, pv4 = pt(R_UP1), pt(R_UP2), pt(R_LO1), pt(R_LO2)
            # 垂直距離
            v1 = _dist(pv1, pv3)
            v2 = _dist(pv2, pv4)
            # 水平距離
            hdist = _dist(p1, p2)
            return _safe_ratio(v1 + v2, 2.0 * hdist)

        ear_l = ear(True)
        ear_r = ear(False)

        # MAR（mouth aspect ratio）：上下唇距 / 嘴角距
        mouth_h = _dist(pt(M_UP), pt(M_LOW))
        mouth_w = _dist(pt(M_LEFT), pt(M_RIGHT))
        mar = _safe_ratio(mouth_h, mouth_w)

        # IPD（interpupillary distance）：左右眼外側角距 / 臉寬近似
        ipd = mouth_w  # 也可用 _dist(pt(L_OUT), pt(R_OUT))
        # 亮度與對比（灰度平均與標準差）
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(gray.mean())
        std_intensity = float(gray.std())

        return np.array([
            ear_l, ear_r, mar, ipd, mean_intensity, std_intensity
        ], dtype=np.float32)
    except Exception:
        return None


def build_features(data_dir: Path, method: str, img_size: int):
    rows = []
    X_list, y_list = [], []

    if method == "landmarks" and not MP_AVAILABLE:
        print("❌ 需要 mediapipe，但目前尚未安裝或載入失敗。請先：pip install mediapipe", file=sys.stderr)
        sys.exit(1)

    face_mesh = None
    if method == "landmarks":
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    for img_path, label in iter_images_by_class(data_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if method == "pixels":
            feat = extract_pixels_feature(img, img_size=img_size)
            feat_names = [f"px_{i}" for i in range(feat.size)]
        else:
            feat = extract_landmarks_feature(img, face_mesh)
            feat_names = ["ear_l", "ear_r", "mar", "ipd", "mean_intensity", "std_intensity"]

        if feat is None:
            # 臉偵測失敗或其他問題
            continue

        X_list.append(feat)
        y_list.append(label)

        row = {"path": str(img_path), "label": label}
        for k, v in zip(feat_names, feat.tolist()):
            row[k] = v
        rows.append(row)

    if face_mesh is not None:
        face_mesh.close()

    if not X_list:
        print("❌ 沒有成功產生任何特徵，請檢查資料夾與圖片內容。", file=sys.stderr)
        sys.exit(1)

    X = np.vstack(X_list)
    y = np.array(y_list)
    df = pd.DataFrame(rows)
    return X, y, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="資料根目錄（底下放各類別子資料夾）")
    ap.add_argument("--method", type=str, default="landmarks", choices=["landmarks", "pixels"],
                    help="特徵萃取方式：landmarks（推薦）或 pixels（基線）")
    ap.add_argument("--img_size", type=int, default=64, help="pixels 方法的縮放邊長")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.0, help="（可選）若想再切驗證集，可在訓練集內再切")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_csv = data_dir / "meta.csv"
    out_model = data_dir / "tabular_model.pkl"

    print(f"📂 掃描資料夾：{data_dir}")
    print(f"🔧 特徵方法：{args.method}")

    X, y, df = build_features(data_dir, method=args.method, img_size=args.img_size)

    # 存 CSV（方便你之後用別的模型重複實驗）
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"📝 已輸出特徵表：{out_csv}（{len(df)} 筆）")

    # 切資料
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # 你也可以加 val_size：在訓練集內再切一份驗證集，這裡先給範例（預設不切）
    if args.val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=args.val_size, random_state=args.seed, stratify=y_train
        )
        print(f"📦 Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    else:
        print(f"📦 Train={len(y_train)}, Test={len(y_test)}")

    # 建 pipeline：標準化 + 邏輯迴歸（你可以很快換 clf）
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=1000, n_jobs=None))
    ])

    print("🚀 開始訓練...")
    clf.fit(X_train, y_train)

    # 評估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Test Accuracy: {acc:.4f}\n")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 存模型
    joblib.dump(clf, out_model)
    print(f"\n💾 已存模型：{out_model}")

    print("\n✅ 完成！你可以直接用 meta.csv 做更多 sklearn 嘗試，或載入 tabular_model.pkl 直接預測。")


if __name__ == "__main__":
    main()

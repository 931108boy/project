# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

DATA = Path("data")
CSV  = DATA/"meta.csv"
CKPT = DATA/"tabular_baseline.pkl"

# 1) 讀資料
df = pd.read_csv(CSV)
df = df[df["label"].isin(["understood","confused"])].copy()

# 2) 特徵欄位（可按需增減）
FEATS = ["Aw","Ca","Hp","B","M","magnitude","direction",
         "ear_l","ear_r","mar","ipd","face_area"]
X = df[FEATS].astype(float).values
y = (df["label"] == "confused").astype(int).values
groups = df["subject"].values  # 以人切分，避免洩漏

# 3) 以 subject 分割 train/val
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(gss.split(X, y, groups))
Xtr, Xva = X[tr_idx], X[va_idx]
ytr, yva = y[tr_idx], y[va_idx]

# 4) 標準化 + 邏輯迴歸（很快的 baseline）
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xva_s = scaler.transform(Xva)

clf = LogisticRegression(max_iter=200, class_weight="balanced")
clf.fit(Xtr_s, ytr)

# 5) 評估
yp = clf.predict(Xva_s)
print(classification_report(yva, yp, target_names=["understood","confused"], digits=3))
print("F1(macro) =", f1_score(yva, yp, average="macro"))
print("混淆矩陣：\n", confusion_matrix(yva, yp))

# 6) 存模型
joblib.dump({"scaler": scaler, "clf": clf, "feats": FEATS}, CKPT)
print("✅ 已存模型：", CKPT)

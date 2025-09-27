# -*- coding: utf-8 -*-
"""
讀取 data/all_features.csv 訓練二分類模型（理解=1 / 困惑=0）
- 同時保存多版本（檔名含 model 與時間戳，不覆蓋）
- 可選交叉驗證 (--cv K)
- 輸出：models/<tag>_<model>_<ts>_scaler.joblib
        models/<tag>_<model>_<ts>_model.joblib
        models/<tag>_<model>_<ts>_feat_meta.json
        models/latest.json（記錄最後一次訓練的三個路徑，方便即時推論直接讀）
"""

import os, json, argparse, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=os.path.join("data", "all_features.csv"),
                    help="資料CSV路徑（預設 data/all_features.csv）")
    ap.add_argument("--model", type=str, default="logreg", choices=["logreg","rf"],
                    help="分類器：logreg（邏輯迴歸）或 rf（隨機森林）")
    ap.add_argument("--test_size", type=float, default=0.2, help="測試集比例，預設 0.2")
    ap.add_argument("--random_state", type=int, default=42, help="隨機種子")
    ap.add_argument("--tag", type=str, default="default", help="自訂標籤（會寫進檔名）")
    ap.add_argument("--cv", type=int, default=0, help="K 折交叉驗證（0=關閉；例如 5 啟用 5-fold）")
    return ap.parse_args()

def load_dataset(csv_path: str):
    if not os.path.exists(csv_path):
        raise SystemExit(f"❌ 找不到資料檔：{csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    if "label" not in df.columns:
        raise SystemExit("❌ CSV 缺少 label 欄位。請確認由蒐集程式寫出的格式。")

    df = df.dropna(subset=["label"])
    feat_cols = [c for c in df.columns if c.startswith(("dx_","dy_","dz_"))]
    if "face_width" in df.columns:  feat_cols.insert(0, "face_width")
    if "face_height" in df.columns: feat_cols.insert(1, "face_height")

    if len(feat_cols) == 0:
        raise SystemExit("❌ 找不到特徵欄位（dx_*, dy_*, dz_*）。")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(int).to_numpy()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"✅ 資料：{len(y)} 筆，特徵維度 {X.shape[1]}")
    uniq, cnt = np.unique(y, return_counts=True)
    print("類別分佈：", dict(zip(uniq.tolist(), cnt.tolist())))
    return X, y, feat_cols

def build_model(kind: str, rs: int):
    if kind == "logreg":
        return LogisticRegression(
            class_weight="balanced",
            solver="saga",
            max_iter=1000,
            n_jobs=-1,
            random_state=rs
        )
    else:
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=rs
        )

def kfold_eval(X, y, feat_cols, kind: str, rs: int, k: int):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)
    accs, f1s, aucs = [], [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])

        clf = build_model(kind, rs)
        clf.fit(Xtr, y[tr])

        y_pred = clf.predict(Xva)
        acc = accuracy_score(y[va], y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y[va], y_pred, average="binary", zero_division=0)

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(Xva)[:, 1]
            auc = roc_auc_score(y[va], y_prob)
        else:
            auc = np.nan

        accs.append(acc); f1s.append(f1); aucs.append(auc)
        print(f"[CV fold {fold}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    print("\n[CV 平均]")
    print(f"Acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
          f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
          f"AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)

    X, y, feat_cols = load_dataset(args.csv)

    if args.cv and args.cv >= 2:
        print(f"\n🔎 進行 {args.cv}-fold 交叉驗證…")
        kfold_eval(X, y, feat_cols, args.model, args.random_state, args.cv)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    clf = build_model(args.model, args.random_state)
    clf.fit(X_trs, y_tr)

    y_pred = clf.predict(X_tes)
    auc = np.nan
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_tes)[:, 1]
        auc = roc_auc_score(y_te, y_prob)

    acc = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_te, y_pred)

    print("\n=== 測試集評估 ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if not np.isnan(auc):
        print(f"ROC-AUC  : {auc:.4f}")
    print("\n混淆矩陣 [[TN FP] [FN TP]]：")
    print(cm)
    print("\n分類報告：")
    from sklearn.metrics import classification_report
    print(classification_report(y_te, y_pred, digits=4))

    # 檔名：<tag>_<model>_<ts>_*.joblib/json
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{args.tag}_{args.model}_{ts}"
    scaler_path = os.path.join("models", f"{stem}_scaler.joblib")
    model_path  = os.path.join("models", f"{stem}_model.joblib")
    meta_path   = os.path.join("models", f"{stem}_feat_meta.json")

    dump(scaler, scaler_path)
    dump(clf, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feat_cols,
                   "model_type": args.model,
                   "n_features": len(feat_cols)}, f, ensure_ascii=False, indent=2)

    # 寫一份 latest.json 指向最新模型（即時推論可直接讀它）
    with open(os.path.join("models", "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"scaler": scaler_path, "model": model_path, "meta": meta_path},
                  f, ensure_ascii=False, indent=2)

    print("\n✅ 已儲存：")
    print(" ", scaler_path)
    print(" ", model_path)
    print(" ", meta_path)
    print("➡ models/latest.json 也已更新（方便即時推論直接讀取）")

if __name__ == "__main__":
    main()

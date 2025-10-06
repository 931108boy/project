# -*- coding: utf-8 -*-
"""
train_tabular_baseline_full.py
- 合併 data/meta.csv 與 data/meta/meta_*.csv
- 只留 understood / confused
- 特徵：優先用指定清單，不在就自動用數值欄
- 以 subject 分割；若只有 1 位受測者 → 退回樣本分割（盡量 stratified）
- 保障兩邊都含兩個類別（若辦得到），必要時自動換 seed / 調整 test_size
- StandardScaler + LogisticRegression(class_weight='balanced')
- 列印詳細統計，存模型與訓練摘要 JSON
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json, joblib, argparse

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PREFERRED_FEATS = [
    "Aw","Ca","Hp","B","M","magnitude","direction",
    "ear_l","ear_r","mar","ipd","face_area"
]

def load_all_meta(data_dir: Path) -> pd.DataFrame:
    meta_dir = data_dir / "meta"
    dfs = []
    if meta_dir.exists():
        for p in sorted(meta_dir.glob("meta_*.csv")):
            try:
                dfs.append(pd.read_csv(p))
            except Exception as e:
                print(f"⚠️ 讀取失敗（略過）{p}: {e}")
    legacy = data_dir / "meta.csv"
    if legacy.exists():
        dfs.append(pd.read_csv(legacy))
    if not dfs:
        raise FileNotFoundError(f"找不到任何 CSV：{meta_dir/'meta_*.csv'} 或 {legacy}")
    return pd.concat(dfs, ignore_index=True)

def pick_features(df: pd.DataFrame) -> list:
    feats = [c for c in PREFERRED_FEATS if c in df.columns]
    if feats:
        return feats
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for bad in ["subject"]:
        if bad in num_cols:
            num_cols.remove(bad)
    return num_cols

def derive_groups(df: pd.DataFrame) -> np.ndarray:
    if "subject" in df.columns:
        return df["subject"].astype(str).values
    if "img" in df.columns:
        return df["img"].astype(str).apply(
            lambda s: Path(s).stem.split("_")[0] if "_" in Path(s).stem else "unknown"
        ).values
    return np.array(["unknown"] * len(df))

def ensure_two_classes(y_tr, y_va, max_tries=0):
    """回傳 True 表示兩邊都有兩個類別；max_tries 保留介面但此函式只做檢查"""
    ok_tr = len(np.unique(y_tr)) >= 2
    ok_va = len(np.unique(y_va)) >= 2
    return ok_tr and ok_va

def robust_split(X, y, groups, test_size=0.2, base_seed=42):
    uniq_groups = np.unique(groups.astype(str))
    n_groups = len(uniq_groups)
    n_samples = len(y)

    if n_samples < 2:
        raise ValueError("樣本數太少，至少需要 2 筆。")

    # 盡量讓 train/val 兩邊都包含兩個類別
    seeds = [base_seed + i for i in range(50)]
    sizes = [test_size] + [max(0.1, min(0.4, test_size + d)) for d in (-0.05, 0.05, -0.1, 0.1)]

    if n_groups >= 2:
        print(f"[Split] 以 subject 分割（{n_groups} 人）")
        for s in seeds:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
            tr_idx, va_idx = next(gss.split(X, y, groups))
            if ensure_two_classes(y[tr_idx], y[va_idx]):
                print(f"  ✓ 使用 seed={s}")
                return tr_idx, va_idx
        # 退一步：接受當前切法（可能一邊只有單類別）
        print("  ⚠️ 無法同時保證兩邊皆含兩類別，採用最後一次切分")
        return tr_idx, va_idx

    # 只有一位受測者 → 樣本分割
    print("[Split] 只有 1 位受測者 → 改用樣本分割")
    classes, counts = np.unique(y, return_counts=True)
    can_strat = (len(classes) > 1) and all(c >= 2 for c in counts)

    for s in seeds:
        for ts in sizes:
            if can_strat:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=ts, random_state=s)
                tr_idx, va_idx = next(sss.split(X, y))
            else:
                tr_idx, va_idx = train_test_split(
                    np.arange(n_samples), test_size=ts, random_state=s, shuffle=True
                )
            if ensure_two_classes(y[tr_idx], y[va_idx]) or not can_strat:
                print(f"  ✓ 使用 seed={s}, test_size={ts}")
                return tr_idx, va_idx

    print("  ⚠️ 無法同時保證兩邊皆含兩類別（資料量/分佈不足），採用最後一次切分")
    return tr_idx, va_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data", help="資料根目錄（含 meta/ 或 meta.csv）")
    ap.add_argument("--out",  type=str, default="data/tabular_baseline.pkl", help="模型輸出路徑")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_pkl  = Path(args.out)

    # 1) 讀資料
    df = load_all_meta(data_dir)
    df = df[df["label"].isin(["understood","confused"])].copy()
    if df.empty:
        raise ValueError("讀到的資料沒有 'understood' 或 'confused'。")

    # 2) 特徵與標籤
    feats = pick_features(df)
    if not feats:
        raise ValueError("找不到可用的數值特徵欄位。")
    X = df[feats].astype(float).values
    y = (df["label"] == "confused").astype(int).values
    groups = derive_groups(df)

    # 統計
    print("=== 資料統計 ===")
    print("樣本數 =", len(df), "；受測者數 =", len(np.unique(groups.astype(str))))
    print("類別分佈（0=understood, 1=confused）：", dict(zip(*np.unique(y, return_counts=True))))
    print("使用特徵：", feats)

    # 3) 分割
    tr_idx, va_idx = robust_split(X, y, groups, test_size=args.test_size, base_seed=args.seed)
    Xtr, Xva, ytr, yva = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

    # 4) 標準化 + LR
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")
    clf.fit(scaler.transform(Xtr), ytr)

    # 5) 評估
    yp = clf.predict(scaler.transform(Xva))
    print("\n=== Validation Report ===")
    print(classification_report(yva, yp, target_names=["understood","confused"], digits=3))
    print("F1(macro) =", f1_score(yva, yp, average="macro"))
    print("Confusion matrix:\n", confusion_matrix(yva, yp))

    # 6) 存模型 + 摘要
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "clf": clf, "feats": feats}, out_pkl)
    print("\n✅ 已存模型：", out_pkl.resolve())

    summary = {
        "data_dir": str(data_dir.resolve()),
        "n_samples": int(len(df)),
        "n_subjects": int(len(np.unique(groups.astype(str)))),
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "features": feats,
        "test_size": args.test_size,
        "seed": args.seed,
        "model_path": str(out_pkl.resolve())
    }
    (out_pkl.parent / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("📝 已寫入訓練摘要：", (out_pkl.parent / "train_summary.json").resolve())

if __name__ == "__main__":
    main()

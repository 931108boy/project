# -*- coding: utf-8 -*-
"""
merge_and_train.py
------------------
功能：
1) 掃描 data/ 資料夾底下符合樣式的檔案（預設：session_*.csv）
2) 自動合併成一份總表 data/all_features.csv
   - 允許不同檔的欄位不完全一致（例如 468 vs 478 點）：會做欄位聯集，缺的補 0
   - 僅保留 label ∈ {0,1} 的列；丟棄缺 label 的列
   - 依 time 排序、移除完全重複列
3) 呼叫 train_affect_classifier.py 進行訓練（可指定 model、tag、cv）

用法：
    python merge_and_train.py
    python merge_and_train.py --model rf --tag wk2 --cv 5
    python merge_and_train.py --pattern "session_2025*.csv"

先決條件：
- 已有 train_affect_classifier.py 與其相依套件（sklearn, pandas, joblib）
- 你的收集腳本會把檔案放在 ./data/ 之下
"""

import os
import glob
import argparse
import subprocess
import sys
import pandas as pd
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data",
                    help="資料夾路徑（預設 data）")
    ap.add_argument("--pattern", type=str, default="session_*.csv",
                    help="要合併的檔名樣式（glob），預設 session_*.csv")
    ap.add_argument("--output", type=str, default=os.path.join("data", "all_features.csv"),
                    help="輸出總表路徑（預設 data/all_features.csv）")
    ap.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"],
                    help="訓練模型種類：logreg 或 rf（預設 logreg）")
    ap.add_argument("--tag", type=str, default="merged",
                    help="模型存檔用的標籤（寫進檔名，預設 merged）")
    ap.add_argument("--cv", type=int, default=0,
                    help="K 折交叉驗證（0=不啟用；例如 5 啟用 5-fold）")
    ap.add_argument("--train_script", type=str, default="train_affect_classifier.py",
                    help="訓練腳本路徑（預設 train_affect_classifier.py）")
    ap.add_argument("--keep_source", action="store_true",
                    help="合併後保留 source 欄（來源檔名），預設不保留")
    return ap.parse_args()

def list_input_files(data_dir: str, pattern: str):
    glob_pattern = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(glob_pattern))
    return [f for f in files if os.path.isfile(f)]

def read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["__source"] = os.path.basename(path)
    return df

def union_concat(dfs):
    """
    欄位做聯集後合併；缺的欄位補 0（除了 label/time/source 保留原樣）
    """
    # 全部欄位聯集
    all_cols = set()
    for d in dfs:
        all_cols.update(d.columns)

    # 統一順序：優先 time, label, face_width, face_height；再 dx_/dy_/dz_；最後其他
    base_order = ["time", "label", "face_width", "face_height"]
    dyn_cols = sorted([c for c in all_cols if c.startswith(("dx_", "dy_", "dz_"))],
                      key=lambda x: (x.split("_")[0], int(x.split("_")[1])))
    other_cols = sorted([c for c in all_cols if c not in set(base_order) | set(dyn_cols) | {"__source"}])

    ordered_cols = base_order + dyn_cols + other_cols + ["__source"]

    # 逐個補齊缺欄位並重排
    fixed = []
    for d in dfs:
        for col in ordered_cols:
            if col not in d.columns:
                # 對於數值特徵補 0；其他欄位補 NaN
                if col.startswith(("dx_","dy_","dz_")) or col in ("face_width","face_height"):
                    d[col] = 0.0
                elif col in ("time","label"):
                    d[col] = np.nan
                else:
                    d[col] = np.nan
        fixed.append(d[ordered_cols])

    out = pd.concat(fixed, axis=0, ignore_index=True)
    return out

def clean_merged(df: pd.DataFrame, keep_source: bool = False) -> pd.DataFrame:
    # 僅保留合法 label
    if "label" in df.columns:
        df = df[df["label"].isin([0,1])]
    else:
        raise SystemExit("❌ 合併後沒有 label 欄位，請檢查來源 CSV。")

    # time 欄位若缺失就補成當前索引（不理想但可工作）
    if "time" not in df.columns:
        df["time"] = np.arange(len(df))
    else:
        # 嘗試轉為浮點
        df["time"] = pd.to_numeric(df["time"], errors="coerce")

    # 依時間排序
    df = df.sort_values("time", ascending=True)

    # 丟掉完全重複列（含特徵欄）
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"ℹ️ 移除完全重複列：{removed} 筆")

    # 清理非數值欄位（除了 time/label/__source），其餘非數值欄嘗試轉 0
    for c in df.columns:
        if c in ("time","label","__source"):
            continue
        if df[c].dtype.kind not in "biufc":  # 非數值
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 若不保留來源欄，刪掉
    if not keep_source and "__source" in df.columns:
        df = df.drop(columns=["__source"])

    # 重設索引
    return df.reset_index(drop=True)

def save_csv(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ 已輸出合併檔：{out_path}（共 {len(df)} 筆）")
    if "label" in df.columns:
        uniq, cnt = np.unique(df["label"].astype(int), return_counts=True)
        print("類別分佈：", dict(zip(uniq.tolist(), cnt.tolist())))

def call_train(train_script: str, csv_path: str, model: str, tag: str, cv: int):
    if not os.path.exists(train_script):
        raise SystemExit(f"❌ 找不到訓練腳本：{train_script}")

    cmd = [
        sys.executable,           # 使用當前 Python
        train_script,
        "--csv", csv_path,
        "--model", model,
        "--tag", tag
    ]
    if cv and cv >= 2:
        cmd += ["--cv", str(cv)]

    print("▶ 開始訓練：", " ".join(cmd))
    # Windows 路徑有空白時，用 list 形式最安全
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"❌ 訓練腳本失敗（exit code {proc.returncode}）")

def main():
    args = parse_args()

    # 1) 列出要合併的檔案
    files = list_input_files(args.data_dir, args.pattern)
    if not files:
        raise SystemExit(f"❌ 找不到符合樣式的檔案：{os.path.join(args.data_dir, args.pattern)}")

    print("🔎 將合併以下檔案：")
    for f in files:
        print("  -", f)

    # 2) 讀入並合併（欄位聯集）
    dfs = [read_one_csv(p) for p in files]
    merged = union_concat(dfs)

    # 3) 清理（保留 label=0/1、排序、去重、補值）
    merged = clean_merged(merged, keep_source=args.keep_source)

    # 4) 輸出 all_features.csv
    save_csv(merged, args.output)

    # 5) 呼叫訓練
    call_train(args.train_script, args.output, args.model, args.tag, args.cv)

if __name__ == "__main__":
    main()

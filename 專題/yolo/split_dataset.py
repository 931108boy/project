# -*- coding: utf-8 -*-
import os, random, shutil
from glob import glob
from itertools import chain

# 根據你的目錄調整這兩行（保持與你現在一致）
ROOT = r"C:\Users\USER\Downloads\yolo\datasets"
SRC  = os.path.join(ROOT, "train")   # 來源只看 train/ 下各類別
VAL  = os.path.join(ROOT, "val")
TEST = os.path.join(ROOT, "test")

VAL_RATIO  = 0.15     # 15% 移到 val
TEST_RATIO = 0.10     # 10% 移到 test
SEED       = 42       # 固定隨機種子，結果可重現
EXTS       = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")

random.seed(SEED)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_images(folder):
    """抓取多種副檔名影像清單"""
    return list(chain.from_iterable(glob(os.path.join(folder, ext)) for ext in EXTS))

def main():
    print(f"📂 ROOT = {ROOT}")
    print(f"📂 SRC(train) = {SRC}")
    print(f"📂 VAL  = {VAL}")
    print(f"📂 TEST = {TEST}")

    # 找類別（train 下的子資料夾）
    if not os.path.isdir(SRC):
        print(f"❌ 找不到資料夾：{SRC}")
        return
    classes = [d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))]
    if not classes:
        print(f"❌ 在 {SRC} 沒有任何類別子資料夾")
        return
    print(f"🔎 類別：{classes}")

    for cls in classes:
        src_cls  = os.path.join(SRC,  cls)
        val_cls  = os.path.join(VAL,  cls)
        test_cls = os.path.join(TEST, cls)
        ensure_dir(val_cls); ensure_dir(test_cls)

        # 只看 train 目前還在的檔案（因此可多次執行而不重複）
        files = list_images(src_cls)
        n = len(files)
        if n == 0:
            print(f"⚠️  {cls}: train 內沒有可移動的檔案，略過")
            continue

        # 計算要移動的數量（向下取整，至少 1 張就會移動）
        n_val  = int(n * VAL_RATIO)
        n_test = int(n * TEST_RATIO)

        # 若圖片很少，避免 0 張時看起來沒動作（可依需求拿掉這段）
        if n == 1:
            n_val, n_test = 0, 0
        elif n == 2:
            n_val, n_test = 1, 0

        random.shuffle(files)
        val_files  = files[:n_val]
        test_files = files[n_val:n_val+n_test]

        print(f"\n[{cls}] train 現有 {n} 張 → 移動：val {len(val_files)}, test {len(test_files)}")
        # 移動到 val
        for f in val_files:
            dst = os.path.join(val_cls, os.path.basename(f))
            if os.path.abspath(f) == os.path.abspath(dst):
                continue  # 理論上不會發生，但保險
            try:
                shutil.move(f, dst)
            except Exception as e:
                print(f"  ❗移動失敗（val）{os.path.basename(f)}: {e}")

        # 移動到 test
        for f in test_files:
            dst = os.path.join(test_cls, os.path.basename(f))
            if os.path.abspath(f) == os.path.abspath(dst):
                continue
            try:
                shutil.move(f, dst)
            except Exception as e:
                print(f"  ❗移動失敗（test）{os.path.basename(f)}: {e}")

        # 最終統計（看移動後 train 還剩多少）
        remaining = len(list_images(src_cls))
        moved_val = len(list_images(val_cls))
        moved_tst = len(list_images(test_cls))
        print(f"   ✅ 完成 {cls}｜現況 → train {remaining}｜val {moved_val}｜test {moved_tst}")

    print("\n🎯 完成。結構如下：")
    print(f"{SRC}\\<class>\\   ← 剩餘即為訓練用")
    print(f"{VAL}\\<class>\\   ← 15% 驗證用")
    print(f"{TEST}\\<class>\\  ← 10% 測試用")

if __name__ == "__main__":
    main()

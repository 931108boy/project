# yolo/split_dataset.py
import os, random, shutil
from glob import glob

BASE = "yolo/datasets"   # 你也可以改成 "datasets"（看你從哪層執行）
SRC  = os.path.join(BASE, "train")
VAL  = os.path.join(BASE, "val")
TEST = os.path.join(BASE, "test")

VAL_RATIO  = 0.15   # 15% 做驗證
TEST_RATIO = 0.10   # 10% 做測試

random.seed(42)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    classes = [d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))]
    for cls in classes:
        files = glob(os.path.join(SRC, cls, "*.jpg"))
        random.shuffle(files)
        n = len(files)
        n_val  = int(n * VAL_RATIO)
        n_test = int(n * TEST_RATIO)
        val_files  = files[:n_val]
        test_files = files[n_val:n_val+n_test]

        for sub, lst in (("val", val_files), ("test", test_files)):
            out_dir = os.path.join(BASE, sub, cls)
            ensure_dir(out_dir)
            for f in lst:
                shutil.copy2(f, os.path.join(out_dir, os.path.basename(f)))

    print("✅ Done. 結構：")
    print(f"{BASE}/train/<class>/")
    print(f"{BASE}/val/<class>/")
    print(f"{BASE}/test/<class>/")

if __name__ == "__main__":
    main()

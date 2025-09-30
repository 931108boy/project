# yolo/train_yolo_classifier.py
from ultralytics import YOLO
from pathlib import Path

def main():
    # 改成你的實際資料夾「絕對路徑」
    data_dir = r"C:\Users\USER\Downloads\yolo\datasets"

    # 也可用 Path 產生與檢查
    data_dir = str(Path(data_dir).resolve())

    model = YOLO("yolov8n-cls.pt")   # 或 "yolo11n-cls.pt"

    print("🔎 Using dataset:", data_dir)  # 看看實際路徑
    results = model.train(
        data=data_dir,
        imgsz=128,
        epochs=50,
        batch=8,
        patience=10,
        lr0=0.003,
        optimizer='SGD',
        seed=42,
        verbose=True,
        project="runs/classify",
        name="train_custom",
    )
    print("訓練完成，最佳權重在：", results.best)

if __name__ == "__main__":
    main()

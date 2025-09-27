# yolo/train_yolo_classifier.py
from ultralytics import YOLO

def main():
    data_dir = "yolo/datasets"   # 改成你的資料夾
    model = YOLO("resnet18")     # 也可 "efficientnet_b0" / "mobilenetv3_small"

    results = model.train(
        data=data_dir,
        imgsz=224,
        epochs=50,
        batch=64,
        patience=10,         # 早停
        lr0=0.003,           # 初始學習率
        optimizer='SGD',     # 或 'Adam' / 'AdamW'
        seed=42,
        verbose=True,
        project="runs/classify",
        name="train_custom",
    )
    print("訓練完成，最佳權重在：", results.best)

if __name__ == "__main__":
    main()

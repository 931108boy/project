# yolo/eval_yolo_classifier.py
from ultralytics import YOLO

def main():
    weights = "runs/classify/train/weights/best.pt"  # 換成你的實際路徑
    data_dir = "yolo/datasets"

    model = YOLO(weights)
    # 驗證集
    results_val = model.val(data=data_dir, imgsz=224)
    print("Val metrics:", results_val.results_dict)

    # 測試集
    results_test = model.val(data=data_dir, imgsz=224, split="test")
    print("Test metrics:", results_test.results_dict)

if __name__ == "__main__":
    main()

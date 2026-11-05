"""Train the plain YOLOv8n baseline with the same settings, for comparison.

Quick test:        python train_baseline.py --epochs 1
Full run:          python train_baseline.py --epochs 50
Pretrained start:  python train_baseline.py --weights yolov8n.pt
"""
import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml", help="dataset yaml (absolute dataset path recommended)")
    ap.add_argument("--weights", default="yolov8n.yaml",
                    help="yolov8n.yaml = train from scratch (default, as in the baseline run); yolov8n.pt = start from COCO-pretrained weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU, or cpu")
    args = ap.parse_args()

    tag = "pretrained" if args.weights.endswith(".pt") else "scratch"
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs",
        name=f"baseline_yolov8n_{tag}_{args.epochs}ep",
        seed=0,
    )
    model.val(data=args.data)

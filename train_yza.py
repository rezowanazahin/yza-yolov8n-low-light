"""Train Y-ZA YOLOv8n (YOLOv8n + ECA + GC after each neck C2f layer) on ExDark.

Quick test:        python train_yza.py --epochs 1
Full run:          python train_yza.py --epochs 50
Pretrained start:  python train_yza.py --weights yolov8n.pt
"""
import argparse

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from attention_modules import add_attention


class YZATrainer(DetectionTrainer):
    """Detection trainer that builds the model with ECA and GC added to the neck."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        add_attention(model)
        return model


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
        trainer=YZATrainer,
        project="runs",
        name=f"yza_yolov8n_{tag}_{args.epochs}ep",
        seed=0,
    )
    model.val(data=args.data)

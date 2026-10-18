"""Train a YOLO model on a Roboflow/Kaggle dataset exported in YOLOv8 format.

Examples:
    python -m src.train --data datasets/helmet/data.yaml --name helmet
    python -m src.train --data datasets/plate/data.yaml  --name plate
The best weights are copied to models/<name>.pt
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to dataset data.yaml")
    ap.add_argument("--name", required=True, help="run/model name, e.g. helmet or plate")
    ap.add_argument("--base", default="yolov8n.pt", help="starting weights (yolov8s.pt = more accurate, slower)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project="runs", name=args.name)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    shutil.copy(best, f"models/{args.name}.pt")
    print(f"\nSaved -> models/{args.name}.pt")

    metrics = YOLO(f"models/{args.name}.pt").val(data=args.data)
    print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}  <- put these in your README")


if __name__ == "__main__":
    main()

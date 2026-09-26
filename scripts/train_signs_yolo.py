#!/usr/bin/env python3
"""Train YOLOv8n on synthetic signs. Run on PC (faster), copy models/signs.pt to Pi."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "signs_yolo", "data.yaml")
OUT = os.path.join(ROOT, "models", "signs.pt")


def main():
    if not os.path.isfile(DATA):
        print("Run first: python scripts/print_signs.py && python scripts/build_sign_yolo_dataset.py")
        sys.exit(1)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA,
        epochs=int(os.environ.get("SIGN_EPOCHS", "60")),
        imgsz=320,
        batch=16,
        project=os.path.join(ROOT, "runs", "signs"),
        name="train",
        exist_ok=True,
    )
    best = os.path.join(ROOT, "runs", "signs", "train", "weights", "best.pt")
    if not os.path.isfile(best):
        print("Training finished but best.pt not found")
        sys.exit(1)
    import shutil

    shutil.copy2(best, OUT)
    print("Saved", OUT)
    print("On Pi: pip install ultralytics && copy models/signs.pt")


if __name__ == "__main__":
    main()

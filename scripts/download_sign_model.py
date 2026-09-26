#!/usr/bin/env python3
"""One-time download of pretrained sign YOLO weights (no training)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from signs_yolo import resolve_yolo_weights


def main():
    print("Preset:", getattr(config, "SIGN_YOLO_PRESET", "road"))
    path = resolve_yolo_weights()
    print("Weights:", path)
    try:
        from ultralytics import YOLO

        YOLO(path)
        print("OK — restart smart-car.service")
    except ImportError:
        print("pip install ultralytics")
        sys.exit(1)


if __name__ == "__main__":
    main()

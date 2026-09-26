#!/usr/bin/env python3
"""Synthetic YOLO dataset from signs/*.png → data/signs_yolo/"""

import os
import random

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNS = os.path.join(ROOT, "signs")
OUT = os.path.join(ROOT, "data", "signs_yolo")

# Class names must match train_signs_yolo.py / parse_yolo_class
CLASSES = ["STOP", "30", "40", "red_light", "green_light"]


def _bg(w, h):
    base = random.randint(70, 120)
    img = np.random.randint(base - 25, base + 25, (h, w, 3), np.uint8)
    if random.random() < 0.35:
        cv2.rectangle(img, (0, int(h * 0.4)), (w, h), (55, 58, 62), -1)
    return img


def _paste(bg, fg, cx, cy, scale):
    fh, fw = fg.shape[:2]
    nw = max(20, int(fw * scale))
    nh = max(20, int(fh * scale))
    s = cv2.resize(fg, (nw, nh), interpolation=cv2.INTER_AREA)
    x1 = cx - nw // 2
    y1 = cy - nh // 2
    x2, y2 = x1 + nw, y1 + nh
    H, W = bg.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None
    if s.shape[2] == 4:
        alpha = s[:, :, 3:4].astype(np.float32) / 255.0
        rgb = s[:, :, :3]
        roi = bg[y1:y2, x1:x2].astype(np.float32)
        bg[y1:y2, x1:x2] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    else:
        bg[y1:y2, x1:x2] = s
    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H
    bw = nw / W
    bh = nh / H
    return xc, yc, bw, bh


def _load_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def main():
    mapping = {
        "STOP": os.path.join(SIGNS, "stop.png"),
        "30": os.path.join(SIGNS, "limit30.png"),
        "40": os.path.join(SIGNS, "limit40.png"),
        "red_light": os.path.join(SIGNS, "red_light.png"),
        "green_light": os.path.join(SIGNS, "green_light.png"),
    }
    assets = {}
    for cls, path in mapping.items():
        img = _load_rgba(path)
        if img is None:
            raise SystemExit("Missing " + path + " — run scripts/print_signs.py")
        assets[cls] = img

    for split, n in (("train", 320), ("val", 80)):
        idir = os.path.join(OUT, "images", split)
        ldir = os.path.join(OUT, "labels", split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        W, H = 424, 240
        for i in range(n):
            bg = _bg(W, H)
            labels = []
            k = random.randint(1, 2)
            for _ in range(k):
                cls = random.choice(CLASSES)
                cx = random.randint(int(W * 0.25), int(W * 0.75))
                cy = random.randint(int(H * 0.08), int(H * 0.45))
                sc = random.uniform(0.12, 0.32)
                box = _paste(bg, assets[cls], cx, cy, sc)
                if box is None:
                    continue
                cid = CLASSES.index(cls)
                labels.append(f"{cid} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
            name = f"{split}_{i:04d}"
            cv2.imwrite(os.path.join(idir, name + ".jpg"), bg)
            with open(os.path.join(ldir, name + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(labels))

    yaml_path = os.path.join(OUT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            f"path: {OUT.replace(chr(92), '/')}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(CLASSES)}\n"
            f"names: {CLASSES}\n"
        )
    print("Dataset:", OUT)
    print("Classes:", CLASSES)


if __name__ == "__main__":
    main()

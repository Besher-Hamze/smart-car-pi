#!/usr/bin/env python3
"""Build signs/ref_*.jpg from PNGs or your calibration photos (5 images in order)."""

import os
import sys

import cv2

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signs")

PAIRS = (
    ("ref_stop.jpg", "stop.png"),
    ("ref_limit30.jpg", "limit30.png"),
    ("ref_limit50.jpg", "limit50.png"),
    ("ref_red_panel.jpg", "red_light.png"),
    ("ref_green_panel.jpg", "green_light.png"),
)

PHOTO_OUTS = (
    "ref_stop.jpg",
    "ref_limit30.jpg",
    "ref_limit50.jpg",
    "ref_red_panel.jpg",
    "ref_green_panel.jpg",
)


def square128(path_in, path_out):
    img = cv2.imread(path_in)
    if img is None:
        return False
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = cv2.resize(img[y0 : y0 + s, x0 : x0 + s], (128, 128), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path_out, crop)
    return True


def main():
    os.makedirs(ROOT, exist_ok=True)
    extras = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(extras) >= 5:
        for path, out in zip(extras[:5], PHOTO_OUTS):
            ok = square128(path, os.path.join(ROOT, out))
            print(out, "OK" if ok else "skip")
        return
    for out, src in PAIRS:
        ok = square128(os.path.join(ROOT, src), os.path.join(ROOT, out))
        print(out, "OK" if ok else "skip (no " + src + ")")


if __name__ == "__main__":
    main()

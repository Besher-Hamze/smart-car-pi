#!/usr/bin/env python3
"""Test sign detector on still images (same ROI/scale as on the Pi)."""

import argparse
import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from signs_cv import SimpleSignDetector


def prep(frame):
    h, w = frame.shape[:2]
    y0 = int(h * float(config.SIGN_ROI_TOP))
    y1 = int(h * float(config.SIGN_ROI_BOTTOM))
    roi = frame[y0:max(y1, y0 + 8), :]
    tw = int(config.SIGN_W)
    th = max(8, int(roi.shape[0] * tw / max(roi.shape[1], 1)))
    return cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description="Run ref-based sign detection on images")
    ap.add_argument("images", nargs="*", help="Image paths (default: signs/photo test set)")
    ap.add_argument("--glob", dest="glob_pat", default="", help="Glob for images")
    args = ap.parse_args()

    paths = list(args.images)
    if args.glob_pat:
        paths.extend(glob.glob(args.glob_pat))
    if not paths:
        assets = os.path.join(
            os.path.dirname(ROOT),
            ".cursor",
            "projects",
            "d-smart-car-pi",
            "assets",
            "*photo_*_2026-09-19*.jpg",
        )
        paths = sorted(glob.glob(assets))
    if not paths:
        print("No images — pass paths or --glob")
        return 1

    det = SimpleSignDetector()
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(p, "SKIP (unreadable)")
            continue
        r = det.scan(prep(img))
        print(os.path.basename(p)[:48], "->", r.get("read_text") or "(none)", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

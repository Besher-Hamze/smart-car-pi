#!/usr/bin/env python3
"""Generate printable PNG signs into ./signs/ (run from repo root)."""

import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "signs")


def stop_sign():
    img = np.zeros((500, 500, 3), np.uint8)
    cv2.circle(img, (250, 250), 220, (20, 20, 220), -1)
    cv2.putText(img, "STOP", (55, 310), cv2.FONT_HERSHEY_DUPLEX, 2.4, (255, 255, 255), 4, cv2.LINE_AA)
    return img


def limit_sign(n):
    img = np.full((500, 500, 3), 255, np.uint8)
    cv2.circle(img, (250, 250), 220, (20, 20, 220), 18)
    text = str(n)
    cv2.putText(
        img,
        text,
        (120 if n == 30 else 135, 320),
        cv2.FONT_HERSHEY_DUPLEX,
        3.2,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    return img


def traffic_light(active):
    img = np.full((520, 200, 4), 0, np.uint8)
    img[:, :, :3] = (45, 45, 45)
    img[:, :, 3] = 255
    off = (35, 35, 35)
    red_on = active == "red"
    grn_on = active == "green"
    cv2.circle(img, (100, 130), 62, (0, 0, 240) if red_on else off, -1)
    cv2.circle(img, (100, 280), 62, (0, 220, 0) if grn_on else off, -1)
    cv2.rectangle(img, (20, 40), (180, 380), (25, 25, 25), 4)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, "stop.png"), stop_sign())
    cv2.imwrite(os.path.join(OUT, "red_light.png"), traffic_light("red"))
    cv2.imwrite(os.path.join(OUT, "green_light.png"), traffic_light("green"))
    for n in (30, 40):
        full = limit_sign(n)
        cv2.imwrite(os.path.join(OUT, f"limit{n}.png"), full)
        g = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        crop = g[int(h * 0.28) : int(h * 0.72), int(w * 0.22) : int(w * 0.78)]
        cv2.imwrite(os.path.join(OUT, f"limit{n}_digits.png"), crop)
    print("Wrote signs to", OUT)


if __name__ == "__main__":
    main()

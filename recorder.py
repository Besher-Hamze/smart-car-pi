"""Save camera frames + stick, Donkeycar-style (~10 fps)."""

import csv
import time
from datetime import datetime

import cv2

from pilot import DATA_DIR


class Recorder:
    def __init__(self):
        self.dir = None
        self.csv = None
        self.fh = None
        self.n = 0
        self.last = 0.0
        self._on = False

    @property
    def on(self):
        return self._on

    def set(self, on):
        if on and not self._on:
            self._open()
        if not on and self._on:
            self._close()
        self._on = bool(on)

    def _open(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.dir = DATA_DIR / name
        self.dir.mkdir()
        self.fh = (self.dir / "labels.csv").open("w", newline="", encoding="utf-8")
        self.csv = csv.writer(self.fh)
        self.csv.writerow(["file", "steer", "throttle"])
        self.n = 0
        self.last = 0.0
        print("[rec] writing", self.dir, flush=True)

    def _close(self):
        if self.fh is not None:
            self.fh.close()
            print("[rec] saved", self.n, "frames in", self.dir, flush=True)
        self.fh = None
        self.csv = None

    def write(self, frame, steer, throttle):
        if not self._on or self.csv is None:
            return
        # Keep frames while the car is actually driving (sticky pad or held keys).
        if abs(throttle) < 0.05 and abs(steer) < 0.05:
            return
        now = time.monotonic()
        if now - self.last < 0.10:
            return
        self.last = now
        self.n += 1
        fname = f"{self.n:05d}.jpg"
        cv2.imwrite(str(self.dir / fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        self.csv.writerow([fname, f"{steer:.3f}", f"{throttle:.3f}"])
        self.fh.flush()
        if self.n == 1 or self.n % 25 == 0:
            print(f"[rec] {self.n} frames", flush=True)

    def close(self):
        if self._on:
            self._close()
        self._on = False

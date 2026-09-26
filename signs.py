"""Sign watcher: OpenCV (cv) or ready-made YOLO (yolo) — see SIGN_DETECTOR in config."""

import time

import cv2

import config
from signs_cv import SimpleSignDetector


def _make_detector():
    mode = str(getattr(config, "SIGN_DETECTOR", "cv")).lower().strip()
    if mode == "yolo":
        from signs_yolo import YoloSignAdapter

        det = YoloSignAdapter()
        if det.ok:
            return det
        print("[signs] YOLO unavailable — falling back to cv", flush=True)
    return SimpleSignDetector()


class SignWatcher:
    def __init__(self):
        self.stop_until = 0.0
        self.limit_cap = None
        self.limit_until = 0.0
        self.tl_red = False
        self.label = ""
        self.box = None
        self.boxes = []
        self._stop_hits = 0
        self._limit_hits = 0
        self._red_hits = 0
        self._green_hits = 0
        self._det = _make_detector()
        self._frame_i = 0
        self._last_scan = None

    def _roi(self, frame):
        h, w = frame.shape[:2]
        y0 = int(h * float(getattr(config, "SIGN_ROI_TOP", 0.02)))
        y1 = int(h * float(getattr(config, "SIGN_ROI_BOTTOM", 0.52)))
        y1 = max(y1, y0 + 8)
        return frame[y0:y1, :], y0

    def _scaled(self, bgr):
        tw = int(getattr(config, "SIGN_W", 180))
        th = max(8, int(bgr.shape[0] * tw / max(bgr.shape[1], 1)))
        return cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)

    def _clamp_speed(self, n):
        return max(float(config.SPEED_MIN), min(float(config.SPEED_MAX), float(n)))

    def update(self, frame):
        now = time.monotonic()
        if not getattr(config, "SIGNS_ENABLED", True):
            return self._state(now)

        roi, y0 = self._roi(frame)
        small = self._scaled(roi)
        every_cfg = float(getattr(config, "SIGN_EVERY", 3))
        every = max(2, int(every_cfg) if every_cfg >= 1 else 4)
        if str(getattr(config, "SIGN_DETECTOR", "cv")).lower() == "yolo":
            every = max(every, int(getattr(config, "SIGN_YOLO_EVERY", 4)))
        self._frame_i += 1
        if self._frame_i % every == 0 or self._last_scan is None:
            self._last_scan = self._det.scan(small)
        else:
            self._last_scan = dict(self._last_scan)

        scan = self._last_scan
        sx = roi.shape[1] / max(small.shape[1], 1)
        sy = roi.shape[0] / max(small.shape[0], 1)
        need = int(getattr(config, "SIGN_CONFIRM_FRAMES", 2))

        saw_stop = scan["stop_box"] is not None
        saw_limit = None
        best_limit_box = None
        if scan["limit"] is not None:
            num, x, y, bw, bh = scan["limit"]
            saw_limit = num
            best_limit_box = (int(x * sx), y0 + int(y * sy), int(bw * sx), int(bh * sy))
        if scan.get("read_text"):
            self.label = scan["read_text"][:16]
        elif not saw_stop and saw_limit is None and not scan["red_tl"] and not scan["green_tl"]:
            if now >= self.stop_until and now >= self.limit_until:
                self.label = ""
                self.box = None

        self._apply_hits(
            need,
            saw_stop,
            saw_limit,
            best_limit_box,
            scan["red_tl"],
            scan["green_tl"],
            now,
            stop_box=scan["stop_box"],
            y0=y0,
            sx=sx,
            sy=sy,
        )
        return self._state(now)

    def _apply_hits(
        self,
        need,
        saw_stop,
        saw_limit,
        limit_box,
        saw_red,
        saw_green,
        now,
        stop_box=None,
        y0=0,
        sx=1.0,
        sy=1.0,
    ):
        if saw_stop:
            self._stop_hits += 1
        else:
            self._stop_hits = max(0, self._stop_hits - 1)
        if saw_limit is not None:
            self._limit_hits += 1
        else:
            self._limit_hits = max(0, self._limit_hits - 1)
        if saw_red:
            self._red_hits += 1
        else:
            self._red_hits = max(0, self._red_hits - 1)
        if saw_green:
            self._green_hits += 1
        else:
            self._green_hits = max(0, self._green_hits - 1)

        if self._stop_hits >= need and saw_stop and saw_limit is None:
            self.limit_cap = None
            self.limit_until = 0.0
            self._limit_hits = 0

        if self._limit_hits >= need and saw_limit is not None:
            self._stop_hits = 0
            self.stop_until = 0.0
            self.tl_red = False
            self.box = limit_box
            cap = self._clamp_speed(getattr(config, f"SIGN_LIMIT_{int(saw_limit)}", saw_limit))
            self.limit_cap = cap
            self.limit_until = now + float(getattr(config, "SIGN_LIMIT_LINGER_SEC", 2.5))
            self.label = str(int(cap))
            self._limit_hits = need
        elif self._stop_hits >= need and saw_stop and saw_limit is None:
            if stop_box is not None:
                x, y, bw, bh = stop_box
                self.box = (int(x * sx), y0 + int(y * sy), int(bw * sx), int(bh * sy))
            self.stop_until = max(
                self.stop_until,
                now + float(getattr(config, "SIGN_STOP_HOLD_SEC", 2.2)),
            )
            self.label = "STOP"
            self._stop_hits = need

        if self._green_hits >= need and saw_green:
            self.tl_red = False
            self.label = "GO"
            self._red_hits = 0
        elif self._red_hits >= need and saw_red and saw_limit is None and not saw_stop:
            self.tl_red = True
            if self.label != "STOP":
                self.label = "RED"

    def _state(self, now):
        halt = (now < self.stop_until) or self.tl_red
        cap = None
        if self.limit_cap is not None and now < self.limit_until:
            cap = self.limit_cap
        elif now >= self.limit_until:
            self.limit_cap = None
        return halt, cap, self.label

    def draw(self, vis, halt, cap, label):
        if self.box:
            x, y, bw, bh = self.box
            col = (0, 0, 255) if label in ("STOP", "RED") else (0, 200, 255)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 2)
        if halt and label == "RED":
            cv2.putText(vis, "RED LIGHT", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif halt:
            cv2.putText(vis, "SIGN STOP", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif cap is not None:
            cv2.putText(vis, f"LIMIT {int(cap)}", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        elif label == "GO":
            cv2.putText(vis, "GREEN GO", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

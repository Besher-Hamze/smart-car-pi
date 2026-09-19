"""Camera only: follow yellow paint (one or two blobs). Stay on the road side of the line.

Works with your track: gray/white road, yellow edge, black outside.
"""

import cv2
import numpy as np

import config


class Follower:
    def __init__(self):
        self.last_off = 0.0
        self.tag = "INIT"
        self.yellow_side = 1  # +1 yellow usually on right in image

    def reset(self):
        self.__init__()

    def _yellow_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array(config.YELLOW_LO, np.uint8), np.array(config.YELLOW_HI, np.uint8))
        b, g, r = cv2.split(bgr)
        bi, gi, ri = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)
        bgr_y = (gi > bi + 4) & (ri > bi + 4) & (gi > 24) & (ri > 24)
        v = hsv[:, :, 2]
        y = ((m > 0) | bgr_y) & (v > 18)
        out = (y.astype(np.uint8) * 255)
        return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def _blobs(self, mask, pw, ph):
        y_min = int(ph * getattr(config, "HUG_Y_MIN", 0.22))
        roi = mask[y_min:, :]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_a = int(getattr(config, "HUG_MIN_AREA", 28))
        blobs = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_a:
                continue
            m = cv2.moments(c)
            if m["m00"] < 1:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"] + y_min
            blobs.append((float(cx), float(cy), float(a)))
        return blobs

    def _pick_one(self, blobs, pw):
        side = str(getattr(config, "YELLOW_LINE_SIDE", "right")).lower()
        blobs = sorted(blobs, key=lambda b: b[0])
        if len(blobs) == 1:
            return blobs[0]
        if side == "left":
            return blobs[0]
        if side == "right":
            return blobs[-1]
        mid = pw * 0.5
        if side == "inner":
            return min(blobs, key=lambda b: abs(b[0] - mid))
        if side == "outer":
            return max(blobs, key=lambda b: abs(b[0] - mid))
        return blobs[-1]

    def _target_x(self, pw, side_key):
        side = str(getattr(config, "YELLOW_LINE_SIDE", "right")).lower()
        if side == "left" or side_key == "left":
            return float(getattr(config, "YELLOW_TARGET_LEFT", 0.36)) * pw
        return float(getattr(config, "YELLOW_TARGET_X", 0.64)) * pw

    def step(self, frame):
        h, w = frame.shape[:2]
        y0 = int(h * config.ROI_TOP)
        roi = frame[y0:, :]
        pw = int(getattr(config, "FOLLOW_W", 200))
        ph = int(getattr(config, "FOLLOW_H", 128))
        small = cv2.resize(roi, (pw, ph), interpolation=cv2.INTER_AREA)
        mask = self._yellow_mask(small)
        blobs = self._blobs(mask, pw, ph)
        vis = frame.copy()
        half = pw * 0.5
        gain = float(getattr(config, "HUG_GAIN", 1.05))
        use_one = int(getattr(config, "YELLOW_LINES", 1)) == 1
        found = False
        pick = None
        target = self._target_x(pw, "right")

        if len(blobs) >= 2 and not use_one:
            blobs.sort(key=lambda b: b[0])
            left, right = blobs[0], blobs[-1]
            cx = 0.5 * (left[0] + right[0])
            err = (cx - half) / half
            self.tag = "2Y"
            self.yellow_side = 1 if right[0] > half else -1
            found = True
            target = half
        elif len(blobs) >= 1:
            pick = self._pick_one(blobs, pw) if len(blobs) >= 2 else blobs[0]
            side_name = str(getattr(config, "YELLOW_LINE_SIDE", "right")).lower()
            sk = "left" if side_name == "left" else "right"
            if side_name in ("inner", "outer"):
                sk = "left" if pick[0] < half else "right"
            target = self._target_x(pw, sk)
            cx, cy, _ = pick
            err = (cx - target) / half
            suf = side_name[:1].upper()
            self.tag = f"HUG-{suf}" if len(blobs) >= 2 else "HUG Y"
            self.yellow_side = -1 if pick[0] < half else 1
            found = True
            blobs = [pick]
        else:
            turn = float(getattr(config, "HUG_SEARCH", 0.55)) * self.yellow_side
            err = turn
            self.tag = "TURN R" if self.yellow_side > 0 else "TURN L"
            found = False

        raw = float(np.clip(err * gain, -1.0, 1.0))
        step = config.STEER_STEP
        if raw > self.last_off + step:
            raw = self.last_off + step
        elif raw < self.last_off - step:
            raw = self.last_off - step
        if abs(raw) < config.STEER_DEAD:
            raw = 0.0
        offset = raw
        self.last_off = offset
        thr = config.AUTO_THR_MIN if found else max(0.4, config.AUTO_THR_MIN * 0.85)
        turn = abs(offset)
        throttle = max(thr, config.AUTO_THR_MIN * (1.0 - config.SLOW_IN_TURN * min(0.75, turn)))

        self._draw(vis, y0, w, h, small, mask, blobs, target, found, pick)
        return found, offset, vis, not found, throttle

    def _draw(self, vis, y0, w, h, small, mask, blobs, target, found, pick=None):
        pw, ph = small.shape[1], small.shape[0]
        sx = (w - 1) / max(pw - 1, 1)
        sy = (h - y0 - 1) / max(ph - 1, 1)
        tx = int(target * sx)
        cv2.line(vis, (tx, y0), (tx, h - 1), (255, 120, 0), 1)
        for cx, cy, _ in blobs:
            r = 9 if pick is not None and abs(cx - pick[0]) < 0.5 else 5
            cv2.circle(vis, (int(cx * sx), y0 + int(cy * sy)), r, (0, 255, 255), 2)
        cv2.line(vis, (w // 2, y0), (w // 2, h - 1), (255, 180, 0), 1)
        col = (0, 255, 0) if found else (0, 180, 255)
        cv2.putText(
            vis,
            f"{self.tag}  {self.last_off:+.2f}",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            col,
            2,
        )

"""Follow the printed gray road with a short memory and look-ahead.

Near scan = where the car is now. Far scan = where the road is going.
Memory keeps the same road strip from frame to frame so a bend is a
direction change, not a new random blob.
"""

import cv2
import numpy as np

import config


def _all_runs(ok):
    if ok.size == 0 or not ok.any():
        return []
    padded = np.concatenate(([False], ok, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _pick_run(ok, w, hint):
    runs = [(a, b) for a, b in _all_runs(ok) if (b - a) >= 0.18 * w]
    if not runs:
        return None
    if hint is None:
        return max(runs, key=lambda r: r[1] - r[0])
    return min(runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - hint))


def _row_road(bgr_row, hint):
    """(left, right, cx) of the remembered road on this row, or None."""
    row = bgr_row.reshape(1, -1, 3)
    hsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)[0]
    s = hsv[:, 1].astype(np.int16)
    v = hsv[:, 2].astype(np.int16)
    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)[0].astype(np.int16)
    w = gray.size
    gray_s = cv2.blur(gray.reshape(1, -1), (1, 15))[0]
    cut = int(min(np.percentile(gray_s, 48), 135))
    road = (gray_s <= cut) & (v < 190) & (s < 90)
    road_u8 = cv2.morphologyEx(
        (road.astype(np.uint8) * 255).reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, 27), np.uint8),
    )[0]
    run = _pick_run(road_u8 > 0, w, hint)
    if run is None:
        return None
    a, b = run
    mid = gray_s[a:b]
    if float(mid.mean()) > 150:
        return None
    if (b - a) > 0.90 * w and float(mid.std()) > 22:
        return None
    dash = (v[a:b] >= 140) & (v[a:b] < 195) & (s[a:b] < 55)
    if int(dash.sum()) >= 4:
        cx = a + float(np.mean(np.flatnonzero(dash)))
    else:
        cx = 0.5 * (a + b)
    if hint is not None and abs(cx - hint) > 0.42 * w:
        return None
    return a, b, float(cx)


class Follower:
    def __init__(self):
        self.near_cx = None
        self.far_cx = None
        self.heading = 0.0
        self.hist = []
        self.lost = 0
        self.ok = 0
        self.reverse = False
        self.curve = 0

    def step(self, frame):
        h, w = frame.shape[:2]
        y_far = int(h * 0.62)
        y_mid = int(h * 0.76)
        y_near = int(h * 0.90)
        y0 = y_far

        near = _row_road(frame[y_near], self.near_cx)
        hint = near[2] if near is not None else self.near_cx
        mid = _row_road(frame[y_mid], hint)
        if mid is not None:
            hint = mid[2]
        far = _row_road(frame[y_far], hint)

        bands = []
        if far is not None:
            bands.append((y_far, far))
        if mid is not None:
            bands.append((y_mid, mid))
        if near is not None:
            bands.append((y_near, near))

        found = near is not None or mid is not None
        if found:
            ncx = near[2] if near is not None else mid[2]
            fcx = far[2] if far is not None else (mid[2] if mid is not None else ncx)
            if self.near_cx is not None:
                ncx = 0.62 * ncx + 0.38 * self.near_cx
            if self.far_cx is not None:
                fcx = 0.62 * fcx + 0.38 * self.far_cx
            self.near_cx = ncx
            self.far_cx = fcx
            self.lost = 0
            self.ok += 1
            if self.ok >= 2:
                self.reverse = False
        else:
            self.ok = 0
            self.lost += 1
            ncx = self.near_cx if self.near_cx is not None else w * 0.5
            fcx = self.far_cx if self.far_cx is not None else ncx
            if self.lost >= config.LOST_REVERSE:
                self.reverse = True

        half = w / 2.0
        near_off = float(np.clip((ncx - half) / half, -1.0, 1.0))
        far_off = float(np.clip((fcx - half) / half, -1.0, 1.0))
        bend = float(np.clip(far_off - near_off, -1.0, 1.0))
        self.heading = 0.55 * bend + 0.45 * self.heading
        self.hist.append(self.heading)
        if len(self.hist) > 8:
            self.hist.pop(0)

        changed = False
        if len(self.hist) >= 4:
            old = float(np.mean(self.hist[:3]))
            new = float(np.mean(self.hist[-3:]))
            if abs(new) > 0.10 and old * new < 0:
                changed = True
            if abs(new - old) > 0.28:
                changed = True

        if abs(self.heading) > 0.16 or abs(far_off) > 0.28 or changed:
            self.curve = 1 if (self.heading + far_off) >= 0 else -1
            tag = "TURN R" if self.curve > 0 else "TURN L"
            boost = 1.35 if changed else 1.20
        else:
            self.curve = 0
            tag = "ROAD"
            boost = 1.0

        offset = (
            config.STEER_GAIN * near_off
            + config.LOOK_GAIN * far_off
            + config.HEAD_GAIN * self.heading
        )
        offset = float(np.clip(offset * boost, -1.0, 1.0))
        if self.reverse:
            tag = "BACK"

        vis = frame.copy()
        cv2.line(vis, (0, y0), (w, y0), (140, 140, 140), 1)
        cv2.line(vis, (w // 2, y0), (w // 2, h), (255, 180, 0), 1)
        pts = []
        for y, (_a, _b, cx) in bands:
            pts.append((int(cx), y))
            cv2.line(vis, (_a, y), (_b, y), (0, 220, 0), 3)
            cv2.circle(vis, (int(cx), y), 6, (0, 255, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 255, 0), 2)
        aim = int(ncx)
        cv2.circle(vis, (aim, y_near), 9, (0, 255, 0) if found else (0, 140, 255), -1)
        strip = frame[y0:, :]
        thumb = cv2.cvtColor(cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        vis[8:98, w - 168 : w - 8] = cv2.resize(thumb, (160, 90), interpolation=cv2.INTER_AREA)
        color = (0, 80, 255) if tag == "BACK" else ((0, 255, 255) if self.curve else (0, 255, 0))
        cv2.putText(
            vis,
            f"{tag} {offset:+.2f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        return found, offset, vis, self.reverse

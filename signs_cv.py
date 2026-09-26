"""Traffic signs: red blob first, then shape — limit (white circle) vs STOP (octagon). No full-frame template."""

import os

import cv2
import numpy as np

import config

PANEL_KINDS = frozenset({"red_tl", "green_tl"})


class SimpleSignDetector:
    def __init__(self):
        root = os.path.join(os.path.dirname(__file__), "signs")
        self._root = root
        self._limit_refs = []
        self._stop_refs = []
        for label, names in (
            ("30", ("ref_limit30.jpg", "limit30.png")),
            ("50", ("ref_limit50.jpg", "limit50.png")),
        ):
            g = self._load_gray(root, names)
            if g is not None:
                self._limit_refs.append((label, g))
        for names in (("ref_stop.jpg", "stop.png"),):
            for name in names:
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if g is not None:
                        self._stop_refs.append(
                            cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
                        )
        self._digit_tmpl = []
        for label, fname in (("30", "limit30_digits.png"), ("50", "limit50_digits.png")):
            path = os.path.join(root, fname)
            if os.path.isfile(path):
                g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    self._digit_tmpl.append((label, g))
        self._panel_refs = []
        for kind, label, names in (
            ("red_tl", "RED", ("ref_red_panel.jpg", "red_light.png")),
            ("green_tl", "GREEN", ("ref_green_panel.jpg", "green_light.png")),
        ):
            g = self._load_gray(root, names)
            if g is not None:
                self._panel_refs.append((kind, label, g))

    @staticmethod
    def _load_gray(root, names):
        refs = [n for n in names if n.startswith("ref_")]
        fall = [n for n in names if not n.startswith("ref_")]
        todo = refs if any(os.path.isfile(os.path.join(root, n)) for n in refs) else fall
        for name in todo:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if g is not None:
                    return cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
        return None

    def scan(self, bgr):
        out = {
            "stop_box": None,
            "limit": None,
            "red_tl": False,
            "green_tl": False,
            "read_text": "",
        }
        h, w = bgr.shape[:2]
        blobs = self._traffic_blobs(bgr, w, h)
        hit = self._best_traffic_blob(bgr, blobs)
        if hit is None and not blobs:
            hit = self._best_panel(bgr, w, h)
        if hit is None:
            return out
        kind, label, box = hit
        out["read_text"] = label
        if kind == "stop":
            out["stop_box"] = box
        elif kind == "limit":
            out["limit"] = (int(label), *box)
        elif kind == "red_tl":
            out["red_tl"] = True
        elif kind == "green_tl":
            out["green_tl"] = True
        return out

    def scan_panels(self, bgr):
        h, w = bgr.shape[:2]
        hit = self._best_panel(bgr, w, h)
        if hit is None:
            return {"red_tl": False, "green_tl": False}
        kind, _, _ = hit
        return {"red_tl": kind == "red_tl", "green_tl": kind == "green_tl"}

    def _red_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (0, 72, 58), (12, 255, 255)) | cv2.inRange(
            hsv, (168, 72, 58), (180, 255, 255)
        )

    def _white_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (0, 0, 168), (180, 62, 255))

    def _traffic_blobs(self, bgr, fw, fh):
        red = self._red_mask(bgr)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        white = self._white_mask(bgr)
        min_a = max(220, int(fw * fh * float(getattr(config, "SIGN_RED_MIN_FRAC", 0.045))))
        cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_a:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 14 or bh < 14:
                continue
            ar = bw / max(bh, 1)
            if ar < 0.45 or ar > 2.2:
                continue
            crop_r = red[y : y + bh, x : x + bw]
            red_fill = float((crop_r > 0).mean())
            if red_fill < float(getattr(config, "SIGN_RED_FILL_MIN", 0.22)):
                continue
            cx, cy = x + bw // 2, y + bh // 2
            r = max(6, int(min(bw, bh) * 0.38))
            wm = np.zeros(white.shape, np.uint8)
            cv2.ellipse(wm, (cx, cy), (max(4, bw // 3), max(4, bh // 3)), 0, 0, 360, 255, -1)
            white_ctr = float(((white > 0) & (wm > 0)).mean())
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.055 * peri, True)
            verts = len(approx)
            circ = 4 * np.pi * a / (peri * peri + 1e-6)
            blobs.append(
                {
                    "box": (x, y, bw, bh),
                    "area": a,
                    "red_fill": red_fill,
                    "white_ctr": white_ctr,
                    "verts": verts,
                    "circ": float(circ),
                    "contour": c,
                }
            )
        blobs.sort(key=lambda b: b["area"], reverse=True)
        return blobs[:4]

    def _best_traffic_blob(self, bgr, blobs):
        fh, fw = bgr.shape[:2]
        hi = float(getattr(config, "SIGN_LIMIT_WHITE_MIN", 0.10))
        lo = float(getattr(config, "SIGN_STOP_WHITE_MAX", 0.08))
        stop_verts_min = int(getattr(config, "SIGN_STOP_VERT_MIN", 6))
        stop_verts_max = int(getattr(config, "SIGN_STOP_VERT_MAX", 11))
        limit_lo = float(getattr(config, "SIGN_LIMIT_WHITE_LO", 0.045))
        best_limit = None
        best_stop = None

        for blob in blobs:
            x, y, bw, bh = blob["box"]
            pad = int(max(bw, bh) * 0.2)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(fw, x + bw + pad)
            y1 = min(fh, y + bh + pad)
            patch = bgr[y0:y1, x0:x1]
            wctr = blob["white_ctr"]

            if wctr >= limit_lo:
                label, score, rel = self._match_limit_patch(patch)
                need = float(getattr(config, "SIGN_LIMIT_MATCH_MIN", 0.30))
                if wctr < hi:
                    need += 0.06
                if label and score >= need:
                    cand = (
                        "limit",
                        label,
                        score,
                        (x0 + rel[0], y0 + rel[1], rel[2], rel[3]),
                    )
                    if best_limit is None or cand[2] > best_limit[2]:
                        best_limit = cand
                if wctr >= hi:
                    continue

            if wctr >= limit_lo:
                continue
            octagon = stop_verts_min <= blob["verts"] <= stop_verts_max
            if wctr <= lo and (octagon or blob["red_fill"] >= 0.32):
                score, rel = self._match_stop_patch(patch)
                if score >= float(getattr(config, "SIGN_STOP_MATCH_MIN", 0.23)):
                    cand = (
                        "stop",
                        "STOP",
                        score,
                        (x0 + rel[0], y0 + rel[1], rel[2], rel[3]),
                    )
                    if best_stop is None or cand[2] > best_stop[2]:
                        best_stop = cand

        if best_limit and best_stop:
            if best_limit[2] >= best_stop[2] + 0.06:
                k, lb, _, box = best_limit
                return k, lb, box
            if best_stop[2] >= best_limit[2] + 0.06:
                k, lb, _, box = best_stop
                return k, lb, box
            return None
        if best_limit:
            k, lb, _, box = best_limit
            return k, lb, box
        if best_stop:
            k, lb, _, box = best_stop
            return k, lb, box
        return None

    def _match_limit_patch(self, patch):
        if not self._limit_refs and not self._digit_tmpl:
            return None, 0.0, (0, 0, patch.shape[1], patch.shape[0])
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        h, w = g.shape[:2]
        best_label, best_sc, best_box = None, 0.0, (0, 0, w, h)
        for label, tmpl in self._limit_refs:
            sc, box = self._template_best(g, tmpl)
            if sc > best_sc:
                best_sc, best_label, best_box = sc, label, box
        digit = self._digit_in_patch(g)
        if digit is not None:
            dl, ds, db = digit
            if ds > best_sc:
                best_sc, best_label, best_box = ds, dl, db
        return best_label, best_sc, best_box

    def _match_stop_patch(self, patch):
        if not self._stop_refs:
            return 0.0, (0, 0, patch.shape[1], patch.shape[0])
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        best = 0.0
        best_box = (0, 0, patch.shape[1], patch.shape[0])
        for tmpl in self._stop_refs:
            sc, box = self._template_best(g, tmpl)
            if sc > best:
                best, best_box = sc, box
        return best, best_box

    def _template_best(self, gray, tmpl):
        h, w = gray.shape[:2]
        best, best_loc, best_wh = 0.0, (0, 0), (w, h)
        for sc in (0.55, 0.72, 0.88, 1.0, 1.15):
            rw = rh = max(14, int(128 * sc))
            if rw >= w or rh >= h:
                rw = min(w - 2, h - 2)
                rh = rw
            if rw < 12 or rh < 12 or rw >= w or rh >= h:
                continue
            t = cv2.resize(tmpl, (rw, rh), interpolation=cv2.INTER_AREA)
            if t.shape[0] > h or t.shape[1] > w:
                continue
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            _, mv, _, loc = cv2.minMaxLoc(res)
            if mv > best:
                best, best_loc, best_wh = float(mv), loc, (rw, rh)
        return best, (best_loc[0], best_loc[1], best_wh[0], best_wh[1])

    def _digit_in_patch(self, gray):
        if not self._digit_tmpl:
            return None
        h, w = gray.shape[:2]
        best = None
        for label, tmpl in self._digit_tmpl:
            for sc in (0.5, 0.75, 1.0, 1.2):
                rw = max(10, int(tmpl.shape[1] * sc))
                rh = max(10, int(tmpl.shape[0] * sc))
                if rw >= w or rh >= h:
                    continue
                t = cv2.resize(tmpl, (rw, rh), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
                _, mv, _, loc = cv2.minMaxLoc(res)
                if best is None or mv > best[1]:
                    best = (label, float(mv), (loc[0], loc[1], rw, rh))
        return best

    def _best_panel(self, bgr, fw, fh):
        solid = self._frame_panel_hint(bgr)
        if solid is None:
            return None
        need = float(getattr(config, "SIGN_PANEL_RED_MIN", 0.38))
        if solid == "green_tl":
            need = float(getattr(config, "SIGN_PANEL_GREEN_MIN", 0.36))
        if not self._panel_solid_frame(bgr, solid, need):
            return None
        label = "RED" if solid == "red_tl" else "GREEN"
        return solid, label, (0, 0, fw, fh)

    def _frame_panel_hint(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red = (
            cv2.inRange(hsv, (0, 60, 50), (12, 255, 255))
            | cv2.inRange(hsv, (168, 60, 50), (180, 255, 255))
        ).mean() / 255.0
        green = cv2.inRange(hsv, (35, 40, 45), (90, 255, 255)).mean() / 255.0
        if green > 0.42 and green > red * 1.35:
            return "green_tl"
        if red > 0.40 and red > green * 1.35:
            return "red_tl"
        return None

    def _panel_solid_frame(self, bgr, kind, need):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if kind == "red_tl":
            m = cv2.inRange(hsv, (0, 60, 50), (12, 255, 255)) | cv2.inRange(
                hsv, (168, 60, 50), (180, 255, 255)
            )
        else:
            m = cv2.inRange(hsv, (35, 40, 45), (90, 255, 255))
        return (m > 0).mean() >= need

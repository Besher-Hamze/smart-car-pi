"""Read sign text (STOP / 30 / 50) with Tesseract + template fallback. Not color-only."""

import os
import re
import shutil

import cv2
import numpy as np

import config


def _tesseract_bin():
    return shutil.which("tesseract")


class SignReader:
    def __init__(self):
        root = os.path.join(os.path.dirname(__file__), "signs")
        self._root = root
        self._glyph_tmpl = self._load_glyph_templates()
        self._has_tesseract = _tesseract_bin() is not None
        if not self._has_tesseract:
            print(
                "Sign OCR: install tesseract for best results —  sudo apt install tesseract-ocr tesseract-ocr-eng",
                flush=True,
            )

    def _load_glyph_templates(self):
        out = {}
        for key, fname in (("STOP", "ref_stop.jpg"), ("30", "ref_limit30.jpg"), ("50", "ref_limit50.jpg")):
            path = os.path.join(self._root, fname)
            if not os.path.isfile(path):
                continue
            g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            out[key] = self._text_band(g)
        return out

    def _text_band(self, gray):
        """High-contrast text band from reference sign photo."""
        g = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(bw) > bw.size * 0.55:
            bw = 255 - bw
        h, w = bw.shape
        return bw[int(h * 0.25) : int(h * 0.78), int(w * 0.12) : int(w * 0.88)]

    def _empty_scan(self):
        return {
            "stop_box": None,
            "limit": None,
            "red_tl": False,
            "green_tl": False,
            "read_text": "",
        }

    def scan_panels(self, bgr):
        return self._traffic_panels(bgr)

    def scan(self, bgr, full=False):
        """full=False: glyph template only (fast). full=True: + Tesseract on center crop."""
        out = self._empty_scan()
        crop, x1, y1 = self._center_crop(bgr)
        kind, val, conf, raw = self._read_crop(crop, ocr=full)
        if kind is not None:
            ch, cw = crop.shape[:2]
            out["read_text"] = raw
            if kind == "stop":
                out["stop_box"] = (x1, y1, cw, ch)
            else:
                out["limit"] = (val, x1, y1, cw, ch)

        if out["stop_box"] is None and out["limit"] is None:
            out.update(self._traffic_panels(bgr))
        return out

    def _center_crop(self, bgr):
        h, w = bgr.shape[:2]
        cw, ch = int(w * 0.68), int(h * 0.58)
        cx, cy = w // 2, int(h * 0.38)
        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        return bgr[y1 : y1 + ch, x1 : x1 + cw], x1, y1

    def _ocr_once(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        scale = min(3.0, max(2.0, 240 / max(w, 1)))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            import pytesseract

            cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=STOP0123456789"
            return pytesseract.image_to_string(otsu, lang="eng", config=cfg)
        except Exception:
            return ""

    def _read_crop(self, crop, ocr=False):
        texts = []
        conf = 0.0
        tmpl_kind, tmpl_val, tmpl_sc = self._match_glyphs(crop)
        if tmpl_kind == "stop" and tmpl_sc >= float(getattr(config, "SIGN_STOP_GLYPH_MIN", 0.52)):
            return "stop", None, tmpl_sc, "STOP"
        if tmpl_kind == "limit" and tmpl_val is not None:
            if tmpl_sc >= float(getattr(config, "SIGN_GLYPH_MIN", 0.48)):
                return "limit", tmpl_val, tmpl_sc, str(tmpl_val)

        if ocr and self._has_tesseract and getattr(config, "SIGN_USE_OCR", True):
            txt = self._ocr_once(crop)
            if txt.strip():
                texts.append(txt)
                conf = 0.55

        if not texts:
            return None, None, 0.0, ""

        kind, val = self._parse_text(" ".join(texts))
        if kind is None:
            return None, None, 0.0, ""
        raw = self._clean_display(texts)
        if conf < 0.25 and tmpl_sc < float(getattr(config, "SIGN_GLYPH_MIN", 0.48)):
            if kind == "limit" and tmpl_sc < float(getattr(config, "SIGN_GLYPH_MIN", 0.48)):
                return None, None, 0.0, raw
            if kind == "stop" and tmpl_sc < float(getattr(config, "SIGN_STOP_GLYPH_MIN", 0.52)):
                return None, None, 0.0, raw
        return kind, val, max(conf, tmpl_sc), raw

    def _match_glyphs(self, crop):
        band = self._text_band(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        best_name, best_sc = None, 0.0
        for name, tmpl in self._glyph_tmpl.items():
            if tmpl.size < 20:
                continue
            sc = self._template_score(band, tmpl)
            if sc > best_sc:
                best_sc = sc
                best_name = name
        if best_name == "STOP" and best_sc >= float(getattr(config, "SIGN_STOP_GLYPH_MIN", 0.52)):
            return "stop", None, best_sc
        if best_name in ("30", "50") and best_sc >= float(getattr(config, "SIGN_GLYPH_MIN", 0.48)):
            return "limit", int(best_name), best_sc
        return None, None, best_sc

    def _template_score(self, band, tmpl):
        if band.size < 30 or tmpl.size < 30:
            return 0.0
        best = 0.0
        for sc in (0.75, 0.95, 1.1):
            th, tw = tmpl.shape[:2]
            rw, rh = max(8, int(tw * sc)), max(8, int(th * sc))
            if rw >= band.shape[1] or rh >= band.shape[0]:
                continue
            t = cv2.resize(tmpl, (rw, rh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED)
            _, mv, _, _ = cv2.minMaxLoc(res)
            best = max(best, mv)
        return best

    def _parse_text(self, text):
        t = re.sub(r"[^A-Z0-9]", "", text.upper())
        t = t.replace("S5OP", "STOP").replace("ST0P", "STOP").replace("5TOP", "STOP")
        if "STOP" in t:
            return "stop", None
        for n in (50, 30, 40, 60, 20):
            if str(n) in t:
                return "limit", n
        m = re.search(r"(\d{2})", t)
        if m:
            return "limit", int(m.group(1))
        return None, None

    def _clean_display(self, texts):
        s = " ".join(texts).strip().upper()
        return s[:24]

    def _traffic_panels(self, bgr):
        """Solid red/green sheets only (large uniform panels)."""
        h, w = bgr.shape[:2]
        band = bgr[: int(h * 0.7), :]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        std = float(np.std(gray))
        if std > float(getattr(config, "SIGN_PANEL_MAX_STD", 42)):
            return {"red_tl": False, "green_tl": False}
        area = band.shape[0] * band.shape[1]
        red = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
        red = cv2.bitwise_or(red, cv2.inRange(hsv, (165, 80, 60), (180, 255, 255)))
        green = cv2.inRange(hsv, (32, 60, 50), (95, 255, 255))
        rf = cv2.countNonZero(red) / max(area, 1)
        gf = cv2.countNonZero(green) / max(area, 1)
        pr = float(getattr(config, "SIGN_PANEL_RED_MIN", 0.38))
        pg = float(getattr(config, "SIGN_PANEL_GREEN_MIN", 0.36))
        red_tl = rf >= pr and gf < 0.1
        green_tl = gf >= pg and rf < 0.12
        return {"red_tl": red_tl and not green_tl, "green_tl": green_tl}

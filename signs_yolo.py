"""Ready-made YOLO weights (no local training): COCO or one-time HF download."""

import os
import re
import urllib.request

import cv2
import numpy as np

import config

_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_yolo_class(name):
    """Map class name → kind: stop | limit | red_light | green_light | traffic_light | speedlimit_box."""
    raw = str(name).strip()
    up = raw.upper().replace("-", "_").replace(" ", "_")
    if up in ("STOP", "STOP_SIGN", "STOP_SIGNAL") or up.startswith("STOP"):
        return "stop", None
    if up in ("RED_LIGHT", "LIGHT_RED", "TL_RED", "TRAFFIC_RED", "RED"):
        return "red_light", None
    if up in ("GREEN_LIGHT", "LIGHT_GREEN", "TL_GREEN", "TRAFFIC_GREEN", "GREEN"):
        return "green_light", None
    if up in ("TRAFFIC_LIGHT", "TRAFFIC_LIGHT_SIGNAL", "TRAFFICLIGHT"):
        return "traffic_light", None
    if "SPEED" in up and "LIMIT" in up or up == "SPEEDLIMIT":
        return "speedlimit_box", None
    if up.isdigit():
        return "limit", int(up)
    m = re.search(r"SPEEDLIMIT[_]?(\d+)", up)
    if m:
        return "limit", int(m.group(1))
    m = re.search(r"(\d{2,3})", up)
    if m and ("SPEED" in up or "LIMIT" in up or "SIGN" in up):
        return "limit", int(m.group(1))
    return None, None


def classify_traffic_light_crop(bgr):
    """Red vs green inside a traffic-light bounding box."""
    if bgr is None or bgr.size < 40:
        return None
    h, w = bgr.shape[:2]
    pad = max(2, int(min(w, h) * 0.08))
    crop = bgr[pad : h - pad, pad : w - pad]
    if crop.size < 20:
        crop = bgr
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    red = cv2.countNonZero(cv2.bitwise_or(red1, red2))
    green = cv2.countNonZero(cv2.inRange(hsv, (35, 50, 50), (95, 255, 255)))
    if red < 25 and green < 25:
        return None
    if red > green * 1.15:
        return "red_light"
    if green > red * 1.15:
        return "green_light"
    return None


def _digit_templates():
    out = {}
    for n in (20, 30, 40, 50, 60):
        img = np.zeros((72, 88), np.uint8)
        cv2.putText(img, str(n), (4, 58), cv2.FONT_HERSHEY_DUPLEX, 1.35, 255, 2, cv2.LINE_AA)
        out[n] = img
    return out


_DIGIT_TMPL = _digit_templates()


def read_speed_from_crop(bgr):
    """Guess speed limit number from sign crop (30, 40, …)."""
    if bgr is None or bgr.size < 60:
        return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    best_n, best_s = None, 0.0
    thr = float(getattr(config, "SIGN_DIGIT_MATCH_MIN", 0.38))
    for n, tmpl in _DIGIT_TMPL.items():
        for sc in (0.5, 0.65, 0.8, 1.0, 1.2):
            th, tw = tmpl.shape[:2]
            rw, rh = max(8, int(tw * sc)), max(8, int(th * sc))
            if rw >= g.shape[1] or rh >= g.shape[0]:
                continue
            t = cv2.resize(tmpl, (rw, rh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
            _, mv, _, _ = cv2.minMaxLoc(res)
            if mv > best_s:
                best_s = mv
                best_n = n
    if best_s >= thr:
        return best_n
    return None


def _download_hf(repo, filename, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 100_000:
        return dest
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=os.path.dirname(dest),
            local_dir_use_symlinks=False,
        )
        if os.path.abspath(path) != os.path.abspath(dest):
            import shutil

            shutil.copy2(path, dest)
        return dest
    except Exception:
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        print(f"[signs] downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dest)
        return dest


def resolve_yolo_weights():
    """Path or hub id for Ultralytics YOLO (downloads automatically when needed)."""
    custom = str(getattr(config, "SIGN_YOLO_MODEL", "") or "").strip()
    if custom:
        path = custom if os.path.isabs(custom) else os.path.join(_ROOT, custom)
        if os.path.isfile(path):
            return path
    preset = str(getattr(config, "SIGN_YOLO_PRESET", "road")).lower()
    if preset in ("coco", "nano", "fast"):
        return "yolov8n.pt"
    dest = os.path.join(_ROOT, "models", "signs_pretrained.pt")
    repo = getattr(config, "SIGN_YOLO_ROAD_REPO", "subhodeepmoitra/Traffic_signals_detection_YOLOv8m")
    file = getattr(config, "SIGN_YOLO_ROAD_WEIGHTS", "best.pt")
    return _download_hf(repo, file, dest)


class YoloSignEngine:
    def __init__(self):
        self.model = None
        self.ok = False
        self.err = ""
        self.weights = ""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.err = "pip install ultralytics"
            return
        try:
            self.weights = resolve_yolo_weights()
            self.model = YOLO(self.weights)
            self.ok = True
            print(f"[signs] YOLO ready: {self.weights}", flush=True)
        except Exception as exc:
            self.err = str(exc)

    def detect(self, bgr):
        if not self.ok:
            return []
        conf = float(getattr(config, "SIGN_YOLO_CONF", 0.38))
        imgsz = int(getattr(config, "SIGN_YOLO_IMGSZ", 320))
        try:
            out = self.model.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)
        except Exception:
            return []
        dets = []
        if not out:
            return dets
        r0 = out[0]
        names = r0.names or getattr(self.model, "names", {})
        if r0.boxes is None:
            return dets
        h, w = bgr.shape[:2]
        for box in r0.boxes:
            cls_id = int(box.cls[0])
            cname = names.get(cls_id, str(cls_id))
            cf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            xi, yi = int(max(0, x1)), int(max(0, y1))
            x2i, y2i = int(min(w, x2)), int(min(h, y2))
            crop = bgr[yi:y2i, xi:x2i]
            kind, val = parse_yolo_class(cname)
            if kind == "traffic_light":
                tl = classify_traffic_light_crop(crop)
                if tl == "red_light":
                    kind, val = "red_light", None
                elif tl == "green_light":
                    kind, val = "green_light", None
                else:
                    continue
            elif kind == "speedlimit_box":
                val = read_speed_from_crop(crop)
                if val is None:
                    val = int(getattr(config, "SIGN_DEFAULT_LIMIT", 40))
                kind = "limit"
            dets.append(
                {
                    "name": cname,
                    "kind": kind,
                    "limit": val,
                    "conf": cf,
                    "x": xi,
                    "y": yi,
                    "w": max(1, x2i - xi),
                    "h": max(1, y2i - yi),
                }
            )
        dets = [d for d in dets if d["kind"] is not None]
        dets.sort(key=lambda d: d["conf"], reverse=True)
        return dets


def _norm_limit(val):
    if val is None:
        return None
    v = int(val)
    for ok in (30, 50):
        if v == ok:
            return ok
    if abs(v - 30) <= 8:
        return 30
    if abs(v - 50) <= 8:
        return 50
    return None


class YoloSignAdapter:
    """Same scan() shape as SimpleSignDetector — for SignWatcher."""

    def __init__(self):
        self._engine = YoloSignEngine()
        self.ok = self._engine.ok

    def scan(self, bgr):
        out = {
            "stop_box": None,
            "limit": None,
            "red_tl": False,
            "green_tl": False,
            "read_text": "",
        }
        if not self._engine.ok:
            return out
        dets = self._engine.detect(bgr)
        if not dets:
            return out
        stop = None
        limit = None
        red = green = False
        for d in dets:
            k = d["kind"]
            if k == "stop" and stop is None:
                stop = d
            elif k == "limit":
                num = _norm_limit(d.get("limit"))
                if num is not None and (limit is None or d["conf"] > limit["conf"]):
                    d = dict(d)
                    d["limit"] = num
                    limit = d
            elif k == "red_light":
                red = True
            elif k == "green_light":
                green = True
        if limit and stop and limit["conf"] >= stop["conf"] + 0.08:
            stop = None
        elif stop and limit and stop["conf"] >= limit["conf"] + 0.08:
            limit = None
        if stop:
            out["stop_box"] = (stop["x"], stop["y"], stop["w"], stop["h"])
            out["read_text"] = "STOP"
        elif limit:
            n = int(limit["limit"])
            out["limit"] = (n, limit["x"], limit["y"], limit["w"], limit["h"])
            out["read_text"] = str(n)
        if red and not green:
            out["red_tl"] = True
            out["read_text"] = "RED"
        elif green and not red:
            out["green_tl"] = True
            out["read_text"] = "GREEN"
        return out

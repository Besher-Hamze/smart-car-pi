"""Donkeycar-style pilot: camera image -> steering (and throttle)."""

from pathlib import Path

import cv2
import numpy as np

import config

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_TFLITE = ROOT / "models" / "pilot.tflite"
MODEL_KERAS = ROOT / "models" / "pilot.keras"
MODEL_PKL = ROOT / "models" / "pilot.pkl"

# Flattened sklearn sizes we have actually trained in this repo.
_SKLEARN_SHAPES = {
    160 * 120 * 3: (config.PILOT_W, config.PILOT_H, 3),
    160 * 120: (160, 120, 1),
    80 * 40 * 3: (80, 40, 3),
    80 * 40: (80, 40, 1),
}


def pack_image(frame, width=None, height=None, channels=3):
    """Lower part of the frame, resized for the net (default 160x120 RGB)."""
    width = int(width or config.PILOT_W)
    height = int(height or config.PILOT_H)
    h = frame.shape[0]
    crop = frame[int(h * 0.35) :, :]
    if channels == 1:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def keys_to_cmd(fwd, back, left, right):
    steer = 0.0
    if left and not right:
        steer = -1.0 if not fwd else -0.65
    elif right and not left:
        steer = 1.0 if not fwd else 0.65
    throttle = 0.0
    if fwd and not back:
        throttle = 1.0
    elif back and not fwd:
        throttle = -1.0
    elif left or right:
        throttle = 0.45
    return steer, throttle


class Pilot:
    def __init__(self, kind, model, width=None, height=None, channels=3):
        self.kind = kind
        self.model = model
        self.width = int(width or config.PILOT_W)
        self.height = int(height or config.PILOT_H)
        self.channels = int(channels)

    @classmethod
    def load(cls):
        if MODEL_TFLITE.is_file():
            try:
                import tflite_runtime.interpreter as tflite
            except Exception:
                try:
                    import tensorflow.lite as tflite
                except Exception:
                    tflite = None
            if tflite is not None:
                it = tflite.Interpreter(model_path=str(MODEL_TFLITE))
                it.allocate_tensors()
                print("[pilot] Donkey CNN (tflite)", MODEL_TFLITE)
                return cls("tflite", it)
        if MODEL_KERAS.is_file():
            try:
                from tensorflow import keras

                model = keras.models.load_model(MODEL_KERAS)
                print("[pilot] Donkey CNN (keras)", MODEL_KERAS)
                return cls("keras", model)
            except Exception as exc:
                print("[pilot] keras load failed:", exc)
        if MODEL_PKL.is_file():
            try:
                import joblib

                model = joblib.load(MODEL_PKL)
                n_in = int(getattr(model, "n_features_in_", 0) or 0)
                shape = _SKLEARN_SHAPES.get(n_in)
                if shape is None:
                    print(
                        f"[pilot] skip old pkl ({n_in} features, need "
                        f"{config.PILOT_W * config.PILOT_H * 3}). Press TRAIN again."
                    )
                    return None
                w, h, c = shape
                kind = "gray" if c == 1 else "RGB"
                print(f"[pilot] sklearn {w}x{h} {kind}  {MODEL_PKL}")
                return cls("sklearn", model, width=w, height=h, channels=c)
            except Exception as exc:
                print("[pilot] pkl load failed:", exc)
        print("[pilot] no model. REC 3 laps, then TRAIN")
        return None

    def act(self, frame):
        img = pack_image(frame, self.width, self.height, self.channels)
        if self.kind == "tflite":
            inp = self.model.get_input_details()[0]
            out = self.model.get_output_details()[0]
            x = img.reshape(1, config.PILOT_H, config.PILOT_W, 3).astype(inp["dtype"])
            self.model.set_tensor(inp["index"], x)
            self.model.invoke()
            y = self.model.get_tensor(out["index"])[0]
        elif self.kind == "keras":
            y = self.model.predict(img[None, ...], verbose=0)[0]
        else:
            y = self.model.predict(img.reshape(1, -1))[0]
            if np.ndim(y) == 0:
                y = np.array([float(y), 1.0])
        steer = float(np.clip(y[0], -1.0, 1.0))
        throttle = float(np.clip(y[1] if len(np.atleast_1d(y)) > 1 else 1.0, 0.0, 1.0))
        return steer, throttle

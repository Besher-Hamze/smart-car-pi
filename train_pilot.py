#!/usr/bin/env python3
"""Train a Donkeycar-style net from REC laps.

CLI:  python3 train_pilot.py
UI:   TRAIN button on the stream page
"""

import csv
import shutil

import cv2
import numpy as np

from pilot import DATA_DIR, MODEL_KERAS, MODEL_PKL, MODEL_TFLITE, pack_image


def count_samples():
    n = 0
    if not DATA_DIR.is_dir():
        return 0
    for run in DATA_DIR.glob("run_*"):
        labels = run / "labels.csv"
        if not labels.is_file():
            continue
        with labels.open(encoding="utf-8") as fh:
            n += max(0, sum(1 for _ in fh) - 1)
    return n


def clear_recordings():
    """Delete every data/run_* folder. Models are kept."""
    n = 0
    if not DATA_DIR.is_dir():
        return 0
    for run in list(DATA_DIR.glob("run_*")):
        if not run.is_dir():
            continue
        shutil.rmtree(run, ignore_errors=True)
        n += 1
    return n


def load_rows():
    rows = []
    runs = sorted(p for p in DATA_DIR.glob("run_*") if (p / "labels.csv").is_file())
    if not runs:
        raise ValueError("ما في تسجيل. اضغط REC وسوق 3 لفات.")
    for run in runs:
        with (run / "labels.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if float(row["throttle"]) < 0.05:
                    continue
                path = run / row["file"]
                if not path.is_file():
                    continue
                rows.append((path, float(row["steer"]), float(row["throttle"])))
    if len(rows) < 80:
        raise ValueError(f"فقط {len(rows)} صورة. سجّل 3 لفات كاملة (هدف 300+).")
    return rows


def augment(img, steer, throttle):
    out = [(img, steer, throttle)]
    b = 0.75 + 0.5 * np.random.random()
    out.append((np.clip(img * b, 0, 1), steer, throttle))
    shift = int(np.random.randint(-6, 7))
    if shift:
        rolled = np.roll(img, shift, axis=1)
        out.append((rolled, float(np.clip(steer + shift * 0.02, -1, 1)), throttle))
    return out


def stack_xy(rows, do_aug):
    xs, ys = [], []
    for path, steer, throttle in rows:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        img = pack_image(bgr)
        pack = augment(img, steer, throttle) if do_aug else [(img, steer, throttle)]
        for im, st, th in pack:
            xs.append(im)
            ys.append([st, th])
    return np.stack(xs), np.array(ys, dtype=np.float32)


def train_keras(x, y, log, epochs=40):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    epochs = max(1, int(epochs))
    inp = keras.Input(shape=(x.shape[1], x.shape[2], 3))
    z = layers.Conv2D(24, 5, strides=2, activation="relu")(inp)
    z = layers.Conv2D(32, 5, strides=2, activation="relu")(z)
    z = layers.Conv2D(48, 5, strides=2, activation="relu")(z)
    z = layers.Conv2D(64, 3, activation="relu")(z)
    z = layers.Flatten()(z)
    z = layers.Dense(100, activation="relu")(z)
    z = layers.Dense(50, activation="relu")(z)
    out = layers.Dense(2)(z)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    class LogCb(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            log(f"epoch {epoch + 1}/{epochs}  loss={logs.get('loss', 0):.4f}")

    log(f"CNN  {epochs} epochs")
    model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.12, callbacks=[LogCb()], verbose=0)
    MODEL_KERAS.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_KERAS)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    MODEL_TFLITE.write_bytes(converter.convert())
    pred = model.predict(x, verbose=0)
    err = float(np.mean(np.abs(pred[:, 0] - y[:, 0])))
    return f"CNN جاهز  {epochs} epoch  خطأ التوجيه={err:.3f}"


def train_sklearn(x, y, log, epochs=40):
    from sklearn.neural_network import MLPRegressor
    import joblib

    epochs = max(20, int(epochs) * 5)
    log(f"tensorflow غير موجود — تدريب شبكة أصغر  max_iter={epochs}")
    xf = x.reshape(len(x), -1)
    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=epochs, random_state=0)
    model.fit(xf, y)
    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PKL)
    pred = model.predict(xf)
    err = float(np.mean(np.abs(pred[:, 0] - y[:, 0])))
    return f"موديل جاهز (sklearn)  خطأ={err:.3f}"


def run_train(log=print, epochs=40):
    """Train from all REC folders. Returns a short status string."""
    log("قراءة التسجيلات...")
    rows = load_rows()
    log(f"{len(rows)} صورة — تدريب {int(epochs)} epoch")
    try:
        x, y = stack_xy(rows, do_aug=True)
        log(f"بعد التعزيز: {len(y)} عينة")
        return train_keras(x, y, log, epochs=epochs)
    except ImportError:
        x0, y0 = stack_xy(rows, do_aug=False)
        return train_sklearn(x0, y0, log, epochs=epochs)


def main():
    print(run_train())
    print("Open AUTO on the stream page — no restart needed if you trained from the UI.")


if __name__ == "__main__":
    main()

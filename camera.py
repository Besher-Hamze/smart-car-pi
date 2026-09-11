"""Raspberry Pi Camera Module 3. Always RGB888 -> BGR so colors stay correct."""

import sys

import cv2

import config


class Camera:
    def __init__(self):
        self._picam = None
        self._cap = None
        if not sys.platform.startswith("win"):
            if self._pi():
                return
        if not self._usb():
            raise SystemExit("No camera. If busy: pkill -f 'python3 main.py' then try again.")

    def _pi(self):
        try:
            from picamera2 import Picamera2
        except Exception:
            return False
        try:
            cam = Picamera2()
            controls = {"FrameRate": config.CAM_FPS}
            try:
                from libcamera import controls as lc

                # Camera Module 3 PDAF spam / stalls if AF keeps hunting.
                controls["AfMode"] = lc.AfModeEnum.Manual
                controls["LensPosition"] = 1.5
            except Exception:
                controls["AfMode"] = 0
                controls["LensPosition"] = 1.5
            raw_kw = {"raw": None}
            try:
                cfg = cam.create_preview_configuration(
                    main={"size": (config.WIDTH, config.HEIGHT), "format": "RGB888"},
                    **raw_kw,
                    controls=controls,
                )
            except Exception:
                cfg = cam.create_preview_configuration(
                    main={"size": (config.WIDTH, config.HEIGHT), "format": "RGB888"},
                    controls=controls,
                )
            cam.configure(cfg)
            cam.start()
            try:
                cam.set_controls(controls)
            except Exception:
                pass
        except Exception as exc:
            print("[camera] busy. Run:  pkill -f 'python3 main.py'   then start again", flush=True)
            print("        ", exc, flush=True)
            raise SystemExit(1) from exc
        self._picam = cam
        print("[camera] Camera Module 3", config.WIDTH, "x", config.HEIGHT, flush=True)
        return True

    def _usb(self):
        cap = cv2.VideoCapture(0)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return False
        self._cap = cap
        print("[camera] webcam", flush=True)
        return True

    def read(self):
        if self._picam is not None:
            try:
                rgb = self._picam.capture_array("main")
            except TypeError:
                rgb = self._picam.capture_array()
            except Exception as exc:
                print("[camera] frame skip:", exc, flush=True)
                return None
            if rgb is None:
                return None
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, frame = self._cap.read()
        if not ok:
            return None
        return cv2.resize(frame, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)

    def close(self):
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

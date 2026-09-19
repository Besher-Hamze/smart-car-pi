"""Raspberry Pi Camera Module 3. OpenCV needs BGR — no extra R/B swap."""

import sys

import cv2

import config


class Camera:
    def __init__(self):
        self._picam = None
        self._cap = None
        self._fmt = "BGR888"
        self._swap_rb = bool(getattr(config, "CAMERA_SWAP_RB", False))
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
            controls = {"FrameRate": config.CAM_FPS, "AwbEnable": True}
            try:
                from libcamera import controls as lc

                controls["AfMode"] = lc.AfModeEnum.Manual
                controls["LensPosition"] = 1.5
                try:
                    controls["AwbMode"] = lc.AwbModeEnum.Indoor
                except Exception:
                    pass
            except Exception:
                controls["AfMode"] = 0
                controls["LensPosition"] = 1.5
            cfg = None
            last_exc = None
            for fmt in ("BGR888", "RGB888"):
                try:
                    kw = dict(
                        main={"size": (config.WIDTH, config.HEIGHT), "format": fmt},
                        controls=controls,
                    )
                    try:
                        cfg = cam.create_preview_configuration(raw=None, **kw)
                    except Exception:
                        cfg = cam.create_preview_configuration(**kw)
                    cam.configure(cfg)
                    self._fmt = fmt
                    break
                except Exception as exc:
                    last_exc = exc
                    cfg = None
            if cfg is None:
                raise last_exc or RuntimeError("camera configure failed")
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
        print(
            f"[camera] Camera Module 3 {config.WIDTH}x{config.HEIGHT} {self._fmt}"
            f"{' swap-RB' if self._swap_rb else ''}",
            flush=True,
        )
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
                frame = self._picam.capture_array("main")
            except TypeError:
                frame = self._picam.capture_array()
            except Exception as exc:
                print("[camera] frame skip:", exc, flush=True)
                return None
            if frame is None:
                return None
            # RGB888 is already OpenCV-BGR on many Pi builds. Extra RGB2BGR
            # swaps yellow into cyan. BGR888 needs no convert. SWAP only if asked.
            if self._fmt == "RGB888" and self._swap_rb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif self._fmt == "BGR888" and self._swap_rb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
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

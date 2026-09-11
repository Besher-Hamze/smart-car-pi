"""HC-SR04 distance in cm. ECHO must be 3.3V (voltage divider), never 5V into the Pi."""

import threading
import time

import config

_HAVE_LGPIO = False
_HAVE_GPIOZERO = False

try:
    import lgpio

    _HAVE_LGPIO = True
except Exception:
    lgpio = None

if not _HAVE_LGPIO:
    try:
        from gpiozero import DigitalInputDevice, DigitalOutputDevice

        _HAVE_GPIOZERO = True
    except Exception:
        DigitalInputDevice = DigitalOutputDevice = None


class Ultrasonic:
    def __init__(self):
        self._cm = None
        self._blocked = False
        self._lock = threading.Lock()
        self._run = False
        self._thread = None
        self._h = None
        self.trig = None
        self.echo = None
        self._backend = None

        if _HAVE_LGPIO:
            try:
                self._h = lgpio.gpiochip_open(0)
                lgpio.gpio_claim_output(self._h, config.PIN_TRIG, 0)
                lgpio.gpio_claim_input(self._h, config.PIN_ECHO)
                self._backend = "lgpio"
            except Exception as exc:
                print("[us] lgpio open failed:", exc)
                if self._h is not None:
                    try:
                        lgpio.gpiochip_close(self._h)
                    except Exception:
                        pass
                    self._h = None
        if self._backend is None and _HAVE_GPIOZERO:
            try:
                self.trig = DigitalOutputDevice(config.PIN_TRIG)
                self.echo = DigitalInputDevice(config.PIN_ECHO, pull_up=None)
                self._backend = "gpiozero"
            except Exception as exc:
                print("[us] GPIO open failed:", exc)
        if self._backend is None:
            print("[us] no GPIO — no distance")
            return

        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(
            f"[us] HC-SR04 {self._backend} TRIG={config.PIN_TRIG} "
            f"ECHO={config.PIN_ECHO}  stop<{config.STOP_CM}cm"
        )

    def _echo_high(self):
        if self._backend == "lgpio":
            return lgpio.gpio_read(self._h, config.PIN_ECHO) == 1
        return bool(self.echo.value)

    def _trig_pulse(self):
        if self._backend == "lgpio":
            lgpio.gpio_write(self._h, config.PIN_TRIG, 0)
            time.sleep(0.00005)
            lgpio.gpio_write(self._h, config.PIN_TRIG, 1)
            time.sleep(0.00001)
            lgpio.gpio_write(self._h, config.PIN_TRIG, 0)
            return
        self.trig.off()
        time.sleep(0.00005)
        self.trig.on()
        time.sleep(0.00001)
        self.trig.off()

    def _ping(self):
        self._trig_pulse()
        deadline = time.perf_counter() + 0.025
        while not self._echo_high():
            if time.perf_counter() > deadline:
                return None
        t1 = time.perf_counter()
        while self._echo_high():
            if time.perf_counter() > deadline:
                return None
        cm = (time.perf_counter() - t1) * 17150.0
        if cm < 2 or cm > 300:
            return None
        return cm

    def _loop(self):
        while self._run:
            hits = []
            try:
                for _ in range(3):
                    if not self._run:
                        break
                    d = self._ping()
                    if d is not None:
                        hits.append(d)
                    time.sleep(0.015)
            except Exception:
                time.sleep(0.2)
                continue
            cm = sorted(hits)[len(hits) // 2] if hits else None
            with self._lock:
                self._cm = cm
                if cm is None:
                    pass
                elif cm <= config.STOP_CM:
                    self._blocked = True
                elif cm >= config.CLEAR_CM:
                    self._blocked = False
            time.sleep(0.04)

    def cm(self):
        with self._lock:
            return self._cm

    def blocked(self):
        with self._lock:
            return self._blocked

    def close(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=0.6)
            self._thread = None
        if self._h is not None and lgpio is not None:
            try:
                lgpio.gpiochip_close(self._h)
            except Exception:
                pass
            self._h = None
        for pin in (self.trig, self.echo):
            if pin is None:
                continue
            try:
                pin.close()
            except Exception:
                pass
        self.trig = self.echo = None

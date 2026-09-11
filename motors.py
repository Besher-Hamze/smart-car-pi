"""L298N: left pair of wheels + right pair of wheels."""

import config

try:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice

    _HAVE_GPIO = True
except Exception:
    _HAVE_GPIO = False


class _Fake:
    value = 0

    def on(self):
        pass

    def off(self):
        pass

    def close(self):
        pass


class Motors:
    def __init__(self):
        if _HAVE_GPIO:
            self.lp = PWMOutputDevice(config.PIN_LEFT_PWM, frequency=1000)
            self.rp = PWMOutputDevice(config.PIN_RIGHT_PWM, frequency=1000)
            self.l1 = DigitalOutputDevice(config.PIN_LEFT_IN1)
            self.l2 = DigitalOutputDevice(config.PIN_LEFT_IN2)
            self.r3 = DigitalOutputDevice(config.PIN_RIGHT_IN3)
            self.r4 = DigitalOutputDevice(config.PIN_RIGHT_IN4)
        else:
            self.lp = self.rp = self.l1 = self.l2 = self.r3 = self.r4 = _Fake()
            print("[motors] no GPIO (PC) — not driving")
        self.stop()

    def _side(self, pwm, a, b, speed, invert):
        if invert:
            speed = -speed
        if speed > 1:
            a.on()
            b.off()
            pwm.value = min(abs(speed) / 100.0, 1.0)
        elif speed < -1:
            a.off()
            b.on()
            pwm.value = min(abs(speed) / 100.0, 1.0)
        else:
            a.off()
            b.off()
            pwm.value = 0

    def drive(self, left, right):
        left = max(-config.MAX_SPEED, min(config.MAX_SPEED, left))
        right = max(-config.MAX_SPEED, min(config.MAX_SPEED, right))
        self._side(self.lp, self.l1, self.l2, left, config.INVERT_LEFT)
        self._side(self.rp, self.r3, self.r4, right, config.INVERT_RIGHT)

    def go(self, offset, throttle=1.0, speed=None):
        """offset -1..+1. throttle 0..1 (Donkey-style). speed is 15..90 from the page."""
        off = max(-1.0, min(1.0, config.STEER_SIGN * float(offset)))
        thr = max(0.25, min(1.0, float(throttle)))
        base = config.SPEED if speed is None else float(speed)
        pwm = base * thr * (1.0 - config.SLOW_IN_TURN * abs(off))
        steer = off * min(config.TURN, max(22.0, base * 1.2))
        left = pwm + steer
        right = pwm - steer
        self.drive(left, right)
        return left, right

    def backup(self, offset, speed=None):
        """Reverse. Steer a little toward the last known line."""
        off = max(-1.0, min(1.0, config.STEER_SIGN * float(offset)))
        base = config.BACKUP_SPEED if speed is None else max(18.0, float(speed) * 0.7)
        steer = off * min(config.TURN, base) * 0.45
        left = -base + steer
        right = -base - steer
        self.drive(left, right)
        return left, right

    def manual(self, fwd, back, left, right, speed=None):
        """WASD / arrows: forward, back, spin, or turn while moving."""
        sp = config.MANUAL_SPEED if speed is None else float(speed)
        turn = min(config.MANUAL_TURN, max(20.0, sp * 1.1))
        throttle = 0
        if fwd and not back:
            throttle = sp
        elif back and not fwd:
            throttle = -sp
        steer = 0
        if right and not left:
            steer = turn
        elif left and not right:
            steer = -turn
        lv = throttle + steer
        rv = throttle - steer
        self.drive(lv, rv)
        return lv, rv

    def stop(self):
        self.drive(0, 0)

    def close(self):
        self.stop()
        for p in (self.lp, self.rp, self.l1, self.l2, self.r3, self.r4):
            try:
                p.close()
            except Exception:
                pass

#!/usr/bin/env python3
"""Drive from a learned model, or teach it with REC, or use the road scanner."""

import signal
import time

import cv2

import config
from camera import Camera
from follow import Follower
from motors import Motors
from pilot import Pilot, keys_to_cmd
from recorder import Recorder
from stream import Control, Hub, start
from ultrasonic import Ultrasonic


def _badge(vis, text, color):
    h, w = vis.shape[:2]
    cv2.putText(vis, text, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def _range_overlay(vis, cm, blocked, backing=False):
    h, w = vis.shape[:2]
    if cm is None:
        label, color = "US -- cm", (180, 180, 180)
    elif blocked and backing:
        label, color = f"BACK  {cm:.0f} cm", (0, 140, 255)
    elif blocked:
        label, color = f"NEAR  {cm:.0f} cm", (0, 165, 255)
    elif cm < 40:
        label, color = f"{cm:.0f} cm", (0, 165, 255)
    else:
        label, color = f"{cm:.0f} cm", (0, 220, 80)
    cv2.putText(vis, label, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    if blocked and backing:
        cv2.rectangle(vis, (0, h // 2 - 32), (w, h // 2 + 32), (0, 0, 160), -1)
        cv2.putText(
            vis,
            label,
            (max(12, w // 2 - 110), h // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )


def _paint(vis, mode, keys, cm, blocked, backing, extra=""):
    if extra:
        color = (0, 0, 255) if extra.startswith("REC") else (0, 255, 255) if "PILOT" in extra else (0, 220, 80)
        _badge(vis, extra, color)
    elif mode == "auto":
        _badge(vis, "AUTO", (0, 220, 80))
    else:
        bits = "".join(
            ch for ch, on in (("F", keys[0]), ("B", keys[1]), ("L", keys[2]), ("R", keys[3])) if on
        ) or "STOP"
        _badge(vis, f"MANUAL {bits}", (0, 180, 255))
    _range_overlay(vis, cm, blocked, backing=backing)
    return vis


def main():
    from train_pilot import clear_recordings, count_samples

    cam = Camera()
    motors = Motors()
    follow = Follower()
    rec = Recorder()
    brain = Pilot.load()
    us = Ultrasonic()
    hub = Hub()
    control = Control()
    control.has_pilot = brain is not None
    control.rec_total = count_samples()
    start(hub, config.STREAM_PORT, control)

    run = True

    def stop(*_):
        nonlocal run
        run = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print("Ready. Page: REC 3 laps then TRAIN then AUTO.")

    n = 0
    last_rec = False
    try:
        while run:
            frame = cam.read()
            if frame is None:
                time.sleep(0.03)
                continue
            mode, fwd, back, left_key, right_key, speed = control.snapshot()
            keys = (fwd, back, left_key, right_key)
            cm = us.cm()
            blocked = us.blocked()
            control.set_range(cm, blocked)

            if control.wipe_rec:
                rec.set(False)
                last_rec = False
                n_runs = clear_recordings()
                with control.lock:
                    control.wipe_rec = False
                    control.rec = False
                    control.rec_n = 0
                    control.rec_total = 0
                control.add_log("REC cleared — " + str(n_runs) + " folders")
            if control.rec != last_rec:
                rec.set(control.rec)
                last_rec = control.rec
                if not rec.on:
                    control.rec_total = count_samples()
                    control.add_log("REC saved  total=" + str(control.rec_total))
            if control.reload_pilot:
                brain = Pilot.load()
                control.has_pilot = brain is not None
                control.reload_pilot = False
                control.add_log("PILOT ready" if brain else "PILOT failed to load")
            if mode == "manual" and rec.on:
                st, th = keys_to_cmd(fwd, back, left_key, right_key)
                rec.write(frame, st, th)
                control.rec_n = rec.n

            if control.train_busy:
                motors.stop()
                vis = frame.copy()
                cv2.putText(
                    vis,
                    control.train_log or "TRAINING",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 180, 80),
                    2,
                )
                hub.update(_paint(vis, "manual", keys, cm, False, False, "TRAINING"))
                n += 1
                continue

            if mode != "auto":
                vis = frame.copy()
                left, right = motors.manual(fwd, back, left_key, right_key, speed=speed)
                extra = f"REC {rec.n}" if rec.on else f"SPD {int(speed)}"
                if rec.on and rec.n == 0:
                    extra = "REC 0  tap fwd"
                hub.update(_paint(vis, mode, keys, cm, False, False, extra))
                n += 1
                continue

            if brain is not None:
                vis = frame.copy()
                try:
                    steer, throttle = brain.act(frame)
                except Exception as exc:
                    print("[pilot] act failed — using road scanner:", exc)
                    brain = None
                    control.has_pilot = False
                    control.add_log("PILOT off — press TRAIN, then AUTO")
                    # fall through to scanline follower
                else:
                    cv2.putText(
                        vis,
                        f"PILOT {steer:+.2f}  thr={throttle:.2f}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    if blocked:
                        left, right = motors.backup(steer, speed=speed)
                        why = "back"
                        backing = True
                    else:
                        left, right = motors.go(steer, throttle, speed=speed)
                        why = "pilot"
                        backing = False
                    hub.update(_paint(vis, "auto", keys, cm, blocked, backing, "PILOT AUTO"))
                    n += 1
                    if n % 15 == 0:
                        dist = f"{cm:5.0f}cm" if cm is not None else "   -- "
                        print(f"{why:8s}  {dist}  st={steer:+.2f}  L={left:.0f} R={right:.0f}")
                    continue

            found, offset, vis, lost_path = follow.step(frame)
            backing = bool(blocked or lost_path or not found)
            extra = "AUTO"

            if backing:
                left, right = motors.backup(offset, speed=speed)
                why = "back"
            else:
                left, right = motors.go(offset, speed=speed)
                why = "road"

            t_end = time.monotonic() + config.STEP_SEC
            while run and time.monotonic() < t_end:
                live = cam.read()
                if live is None:
                    time.sleep(0.02)
                    continue
                found, offset, vis, lost_path = follow.step(live)
                cm = us.cm()
                blocked = us.blocked()
                control.set_range(cm, blocked)
                if control.snapshot()[0] != "auto":
                    break
                if (not backing) and blocked:
                    motors.stop()
                    break
                hub.update(_paint(vis, "auto", keys, cm, blocked, backing, extra))
            motors.stop()

            t_end = time.monotonic() + config.PAUSE_SEC
            while run and time.monotonic() < t_end:
                live = cam.read()
                if live is None:
                    time.sleep(0.02)
                    continue
                found, offset, vis, _r = follow.step(live)
                cm = us.cm()
                blocked = us.blocked()
                control.set_range(cm, blocked)
                hub.update(_paint(vis, "auto", keys, cm, blocked, False, extra))
                if control.snapshot()[0] != "auto":
                    break

            n += 1
            dist = f"{cm:5.0f}cm" if cm is not None else "   -- "
            print(f"{why:8s}  {dist}  off={offset:+.2f}  L={left:.0f} R={right:.0f}")
    finally:
        rec.close()
        motors.close()
        us.close()
        cam.close()
        print("stopped")


if __name__ == "__main__":
    main()

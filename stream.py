"""Live view + Auto / Manual drive: http://PI-IP:8080/"""

import json
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import config

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Smart Car</title>
<style>
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body { margin: 0; background: #0e1116; color: #e8eaed; font-family: system-ui, sans-serif; }
  .wrap { max-width: 920px; margin: 0 auto; padding: 10px 10px 24px; }
  img { width: 100%; border-radius: 10px; background: #000; display: block; }
  .row { display: flex; gap: 8px; margin: 12px 0; align-items: center; flex-wrap: wrap; }
  button {
    border: 0; border-radius: 10px; padding: 14px 18px; font-size: 16px; font-weight: 700;
    cursor: pointer; color: #fff; background: #2a3140;
  }
  button.on { background: #1f8a4c; }
  button.man.on { background: #c47a12; }
  button.rec.on { background: #c62828; }
  button.train.on { background: #5b4db8; }
  button.wipe { background: #6b2a33; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  #logbox {
    margin-top: 12px; border-radius: 10px; background: #0b0f14; border: 1px solid #30363d;
    overflow: hidden;
  }
  #logbox h3 {
    margin: 0; padding: 8px 12px; font-size: 12px; letter-spacing: .08em;
    background: #161b22; color: #8b9cb3; font-weight: 700;
  }
  #log {
    margin: 0; padding: 10px 12px; height: 220px; overflow: auto;
    color: #d3f2c4; font-size: 13px; font-family: ui-monospace, Consolas, monospace;
    white-space: pre-wrap; line-height: 1.45;
  }
  #st { opacity: .85; font-size: 14px; }
  #dist { margin-left: auto; font-size: 22px; font-weight: 800; letter-spacing: .02em; }
  #dist.warn { color: #ffb347; }
  #dist.danger { color: #ff5c5c; }
  .pad {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; max-width: 320px; margin: 8px auto 0;
  }
  .pad button { min-height: 64px; font-size: 22px; }
  .pad button.held { background: #3d6fd8; }
  .hint { text-align: center; opacity: .55; font-size: 13px; margin-top: 12px; }
  .speed {
    display: flex; align-items: center; gap: 10px; margin: 10px 0 4px;
    background: #161b22; border-radius: 10px; padding: 10px 12px;
  }
  .speed label { font-size: 14px; font-weight: 700; white-space: nowrap; }
  .speed input { flex: 1; accent-color: #3d6fd8; }
  #spdval, #epval { color: #7ec8ff; min-width: 2.2em; display: inline-block; text-align: right; }
</style>
</head>
<body>
<div class="wrap">
  <img id="cam" src="/stream" alt="camera">
  <div class="row">
    <button id="auto" class="on">AUTO</button>
    <button id="manual" class="man">MANUAL</button>
    <button id="rec" class="rec">REC</button>
    <button id="train" class="train">TRAIN</button>
    <button id="wipe" class="wipe">مسح REC</button>
    <span id="st">AUTO drive</span>
    <span id="dist">-- cm</span>
  </div>
  <div class="speed">
    <label>SPEED <span id="spdval">40</span></label>
    <input id="spd" type="range" min="15" max="90" value="40">
  </div>
  <div class="speed">
    <label>EPOCHS <span id="epval">40</span></label>
    <input id="ep" type="range" min="1" max="200" value="40">
  </div>
  <div class="pad">
    <div></div>
    <button type="button" data-k="fwd">&#9650;</button>
    <div></div>
    <button type="button" data-k="left">&#9664;</button>
    <button type="button" data-k="stop">&#9632;</button>
    <button type="button" data-k="right">&#9654;</button>
    <div></div>
    <button type="button" data-k="back">&#9660;</button>
    <div></div>
  </div>
  <p class="hint">الأزرار ضغطة واحدة تمشي — ■ توقف. REC لازم العداد يزيد وأنت تسوق. TRAIN بعد 3 لفات.</p>
  <div id="logbox">
    <h3>smart-car.service  ·  journalctl</h3>
    <pre id="log">waiting...</pre>
  </div>
</div>
<script>
const keys = {fwd:false, back:false, left:false, right:false};
let mode = "auto";
let rec = false;
let recN = 0;
let recTotal = 0;
let trainBusy = false;
let hasPilot = false;
let speed = 40;
let epochs = 40;
const map = {w:"fwd", ArrowUp:"fwd", s:"back", ArrowDown:"back", a:"left", ArrowLeft:"left", d:"right", ArrowRight:"right"};

function paint() {
  document.getElementById("auto").classList.toggle("on", mode === "auto");
  document.getElementById("manual").classList.toggle("on", mode === "manual");
  document.getElementById("rec").classList.toggle("on", rec);
  document.getElementById("train").classList.toggle("on", trainBusy);
  document.getElementById("train").disabled = trainBusy;
  document.getElementById("rec").disabled = trainBusy;
  document.getElementById("wipe").disabled = trainBusy;
  document.getElementById("ep").disabled = trainBusy;
  let st = mode === "auto" ? "AUTO" : "MANUAL";
  if (rec) st = "REC " + recN;
  if (trainBusy) st = "TRAINING";
  if (hasPilot) st += "  ·  PILOT";
  st += "  ·  " + recTotal + " صور";
  st += "  ·  " + speed;
  document.getElementById("st").textContent = st;
  const sv = document.getElementById("spdval");
  const sl = document.getElementById("spd");
  if (sv) sv.textContent = String(speed);
  if (sl && document.activeElement !== sl) sl.value = String(speed);
  const ev = document.getElementById("epval");
  const el = document.getElementById("ep");
  if (ev) ev.textContent = String(epochs);
  if (el && document.activeElement !== el) el.value = String(epochs);
  document.querySelectorAll(".pad [data-k]").forEach(b => {
    const k = b.getAttribute("data-k");
    b.classList.toggle("held", k !== "stop" && !!keys[k]);
  });
}

function paintDist(s) {
  const el = document.getElementById("dist");
  el.classList.remove("warn", "danger");
  if (s.cm == null) {
    el.textContent = "-- cm";
    return;
  }
  const n = Math.round(s.cm);
  if (s.blocked) {
    el.textContent = "STOP  " + n + " cm";
    el.classList.add("danger");
  } else {
    el.textContent = n + " cm";
    if (n < 40) el.classList.add("warn");
  }
}

async function post(url, body) {
  try {
    await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
  } catch (e) {}
}

function setMode(m) {
  mode = m;
  if (m === "auto") {
    keys.fwd = keys.back = keys.left = keys.right = false;
    rec = false;
    post("/api/rec", {on: false});
  }
  paint();
  post("/api/mode", {mode: m});
}

document.getElementById("rec").onclick = () => {
  if (trainBusy) return;
  rec = !rec;
  if (rec) setMode("manual");
  post("/api/rec", {on: rec});
  paint();
};

document.getElementById("train").onclick = () => {
  if (trainBusy) return;
  rec = false;
  setMode("manual");
  post("/api/rec", {on: false});
  post("/api/train", {epochs: epochs});
  trainBusy = true;
  paint();
};

document.getElementById("wipe").onclick = () => {
  if (trainBusy) return;
  if (!confirm("مسح كل تسجيلات REC والبدء من صفر؟")) return;
  rec = false;
  recN = 0;
  recTotal = 0;
  post("/api/clear", {});
  paint();
};

function sendKeys() {
  if (mode !== "manual") return;
  post("/api/keys", keys);
}

function tapPad(k) {
  if (k === "stop") {
    keys.fwd = keys.back = keys.left = keys.right = false;
    if (mode !== "manual") setMode("manual");
    sendKeys();
    paint();
    return;
  }
  if (mode !== "manual") setMode("manual");
  keys[k] = !keys[k];
  if (k === "fwd" && keys.fwd) keys.back = false;
  if (k === "back" && keys.back) keys.fwd = false;
  if (k === "left" && keys.left) keys.right = false;
  if (k === "right" && keys.right) keys.left = false;
  sendKeys();
  paint();
}

function hold(k, on) {
  if (k === "stop") {
    keys.fwd = keys.back = keys.left = keys.right = false;
    if (mode !== "manual") setMode("manual");
    else sendKeys();
    paint();
    return;
  }
  if (mode !== "manual") setMode("manual");
  keys[k] = on;
  if (k === "fwd" && on) keys.back = false;
  if (k === "back" && on) keys.fwd = false;
  if (k === "left" && on) keys.right = false;
  if (k === "right" && on) keys.left = false;
  sendKeys();
  paint();
}

document.getElementById("auto").onclick = () => setMode("auto");
document.getElementById("manual").onclick = () => setMode("manual");

document.querySelectorAll(".pad [data-k]").forEach(b => {
  const k = b.getAttribute("data-k");
  b.addEventListener("pointerdown", ev => {
    ev.preventDefault();
    tapPad(k);
  });
});

document.getElementById("spd").addEventListener("input", () => {
  speed = Number(document.getElementById("spd").value);
  document.getElementById("spdval").textContent = String(speed);
  post("/api/speed", {speed: speed});
  paint();
});

document.getElementById("ep").addEventListener("input", () => {
  epochs = Number(document.getElementById("ep").value);
  document.getElementById("epval").textContent = String(epochs);
  post("/api/epochs", {epochs: epochs});
  paint();
});

window.addEventListener("keydown", ev => {
  if (ev.repeat) return;
  if (ev.key === "1") { setMode("auto"); return; }
  if (ev.key === "2") { setMode("manual"); return; }
  if (ev.key === " " ) { ev.preventDefault(); hold("stop", true); return; }
  const k = map[ev.key];
  if (!k) return;
  ev.preventDefault();
  hold(k, true);
});
window.addEventListener("keyup", ev => {
  const k = map[ev.key];
  if (!k) return;
  ev.preventDefault();
  hold(k, false);
});
window.addEventListener("blur", () => {
  keys.fwd = keys.back = keys.left = keys.right = false;
  sendKeys();
  paint();
});

setInterval(() => { if (mode === "manual") sendKeys(); }, 200);

function pullStatus() {
  fetch("/api/status").then(r => r.json()).then(s => {
    if (s.mode) mode = s.mode;
    rec = !!s.rec;
    recN = s.rec_n || 0;
    recTotal = s.rec_total || 0;
    trainBusy = !!s.train_busy;
    hasPilot = !!s.has_pilot;
    if (typeof s.speed === "number") speed = s.speed;
    if (typeof s.epochs === "number") epochs = s.epochs;
    paint();
    paintDist(s);
  }).catch(() => {});
}
pullStatus();
setInterval(pullStatus, 250);

function pullJournal() {
  fetch("/api/logs").then(r => r.json()).then(s => {
    const pre = document.getElementById("log");
    if (!pre) return;
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 48;
    pre.textContent = s.text || "(empty)";
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }).catch(() => {});
}
pullJournal();
setInterval(pullJournal, 1000);
</script>
</body>
</html>
""".encode("utf-8")


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "raspberrypi.local"


class Hub:
    def __init__(self):
        self.jpg = b""
        self.lock = threading.Lock()
        blank = np.zeros((240, 320, 3), np.uint8)
        self.update(blank)

    def update(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not ok:
            return
        with self.lock:
            self.jpg = buf.tobytes()

    def get(self):
        with self.lock:
            return self.jpg


class Control:
    """Shared Auto / Manual state between the browser and the drive loop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "auto"
        self.fwd = False
        self.back = False
        self.left = False
        self.right = False
        self.cm = None
        self.blocked = False
        self.rec = False
        self.rec_n = 0
        self.rec_total = 0
        self.has_pilot = False
        self.train_busy = False
        self.train_log = ""
        self.logs = []
        self.reload_pilot = False
        self.wipe_rec = False
        self.speed = float(config.MANUAL_SPEED)
        self.epochs = int(config.TRAIN_EPOCHS)
        self.touched = time.monotonic()
        self.add_log("ready  —  REC 3 laps, then TRAIN, then AUTO")

    def set_mode(self, mode):
        mode = "manual" if mode == "manual" else "auto"
        with self.lock:
            self.mode = mode
            if mode == "auto":
                self.fwd = self.back = self.left = self.right = False
                self.rec = False
            self.touched = time.monotonic()
        self.add_log("mode " + self.mode)
        return self.mode

    def add_log(self, msg):
        line = time.strftime("%H:%M:%S") + "  " + str(msg)
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > 250:
                self.logs = self.logs[-250:]
            self.train_log = str(msg)
        print("[log]", line, flush=True)

    def set_rec(self, on):
        with self.lock:
            if self.train_busy:
                return False
            self.rec = bool(on)
            if self.rec:
                self.mode = "manual"
                self.rec_n = 0
            rec_on = self.rec
        self.add_log("REC ON" if rec_on else "REC OFF")
        return rec_on

    def log_train(self, msg):
        self.add_log(msg)

    def request_clear(self):
        with self.lock:
            if self.train_busy:
                return False
            self.rec = False
            self.rec_n = 0
            self.wipe_rec = True
        self.add_log("REC clear requested")
        return True

    def start_train(self):
        with self.lock:
            if self.train_busy:
                return False
            self.train_busy = True
            self.rec = False
            self.mode = "manual"
            self.fwd = self.back = self.left = self.right = False
            self.train_log = "training..."
            epochs = self.epochs
        self.add_log("TRAIN started  epochs=" + str(epochs))
        threading.Thread(target=_run_train_job, args=(self,), daemon=True).start()
        return True

    def set_speed(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = config.MANUAL_SPEED
        value = max(config.SPEED_MIN, min(config.SPEED_MAX, value))
        with self.lock:
            self.speed = value
        return value

    def set_epochs(self, value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = config.TRAIN_EPOCHS
        value = max(config.TRAIN_EPOCHS_MIN, min(config.TRAIN_EPOCHS_MAX, value))
        with self.lock:
            self.epochs = value
        return value

    def set_keys(self, data):
        with self.lock:
            self.mode = "manual"
            self.fwd = bool(data.get("fwd"))
            self.back = bool(data.get("back"))
            self.left = bool(data.get("left"))
            self.right = bool(data.get("right"))
            self.touched = time.monotonic()

    def set_range(self, cm, blocked):
        with self.lock:
            self.cm = None if cm is None else float(cm)
            self.blocked = bool(blocked)

    def snapshot(self):
        with self.lock:
            fwd, back, left, right = self.fwd, self.back, self.left, self.right
            mode = self.mode
            speed = self.speed
            rec = self.rec
            stale = (time.monotonic() - self.touched) > (2.8 if rec else config.HOLD_TIMEOUT)
        if mode == "manual" and stale:
            fwd = back = left = right = False
        return mode, fwd, back, left, right, speed

    def status(self):
        mode, fwd, back, left, right, speed = self.snapshot()
        with self.lock:
            cm, blocked = self.cm, self.blocked
            rec, rec_n, rec_total = self.rec, self.rec_n, self.rec_total
            has_pilot = self.has_pilot
            train_busy, train_log = self.train_busy, self.train_log
            epochs = self.epochs
            logs = list(self.logs[-120:])
        return {
            "mode": mode,
            "fwd": fwd,
            "back": back,
            "left": left,
            "right": right,
            "speed": speed,
            "epochs": epochs,
            "cm": cm,
            "blocked": blocked,
            "rec": rec,
            "rec_n": rec_n,
            "rec_total": rec_total,
            "has_pilot": has_pilot,
            "train_busy": train_busy,
            "train_log": train_log,
            "logs": logs,
        }


def _run_train_job(control):
    import time as _time

    from train_pilot import count_samples, run_train

    _time.sleep(0.6)
    with control.lock:
        epochs = control.epochs

    try:
        msg = run_train(log=control.log_train, epochs=epochs)
        control.log_train(msg)
        with control.lock:
            control.has_pilot = True
            control.reload_pilot = True
            control.rec_total = count_samples()
    except Exception as exc:
        control.log_train(str(exc))
        with control.lock:
            control.has_pilot = False
    finally:
        with control.lock:
            control.train_busy = False


_JOURNAL_CACHE = {"t": 0.0, "text": ""}


def read_service_logs(lines=120):
    """Last lines of systemd unit smart-car (same as: journalctl -u smart-car)."""
    now = time.monotonic()
    if now - _JOURNAL_CACHE["t"] < 0.8:
        return _JOURNAL_CACHE["text"]

    cmds = [
        ["journalctl", "-u", "smart-car", "-n", str(lines), "--no-pager", "-o", "short-iso"],
        ["journalctl", "-u", "smart-car.service", "-n", str(lines), "--no-pager", "-o", "short-iso"],
    ]
    text = ""
    err = ""
    for cmd in cmds:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        except Exception as exc:
            err = str(exc)
            continue
        if p.stdout and p.stdout.strip():
            text = p.stdout.strip()
            break
        err = (p.stderr or "").strip() or err
    if not text:
        text = err or "no journal output for smart-car"
        text += "\n\nIf empty:  sudo usermod -aG systemd-journal pi   then reboot"
    _JOURNAL_CACHE["t"] = now
    _JOURNAL_CACHE["text"] = text
    return text


def _read_json(handler):
    n = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(n) if n else b"{}"
    try:
        return json.loads(raw.decode("utf-8") or "{}")
    except Exception:
        return {}


def _send(handler, code, body, ctype):
    data = body if isinstance(body, bytes) else body.encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def start(hub, port, control):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                _send(self, 200, PAGE, "text/html; charset=utf-8")
                return
            if path == "/api/status":
                _send(self, 200, json.dumps(control.status()), "application/json")
                return
            if path == "/api/logs":
                _send(self, 200, json.dumps({"text": read_service_logs()}), "application/json")
                return
            if path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    j = hub.get()
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            data = _read_json(self)
            if path == "/api/mode":
                mode = control.set_mode(data.get("mode", "auto"))
                _send(self, 200, json.dumps({"ok": True, "mode": mode}), "application/json")
                return
            if path == "/api/keys":
                control.set_keys(data)
                _send(self, 200, json.dumps({"ok": True}), "application/json")
                return
            if path == "/api/speed":
                sp = control.set_speed(data.get("speed", config.MANUAL_SPEED))
                _send(self, 200, json.dumps({"ok": True, "speed": sp}), "application/json")
                return
            if path == "/api/epochs":
                ep = control.set_epochs(data.get("epochs", config.TRAIN_EPOCHS))
                _send(self, 200, json.dumps({"ok": True, "epochs": ep}), "application/json")
                return
            if path == "/api/rec":
                on = control.set_rec(bool(data.get("on")))
                _send(self, 200, json.dumps({"ok": True, "rec": on}), "application/json")
                return
            if path == "/api/train":
                if "epochs" in data:
                    control.set_epochs(data.get("epochs"))
                ok = control.start_train()
                _send(
                    self,
                    200,
                    json.dumps({"ok": ok, "busy": control.train_busy}),
                    "application/json",
                )
                return
            if path == "/api/clear":
                ok = control.request_clear()
                _send(self, 200, json.dumps({"ok": ok}), "application/json")
                return
            self.send_error(404)

    httpd = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Open on laptop:  http://{lan_ip()}:{port}/")
    return httpd

#!/bin/bash
# Wait for camera and GPIO after power-on, then start the car.
set -e
cd /home/pi/smart-car-pi
sleep 8

MARKER=.venv/pyvenv.cfg
need_venv=0
if [ ! -f .venv/bin/python3 ]; then
  need_venv=1
elif ! grep -q 'include-system-site-packages = true' "$MARKER" 2>/dev/null; then
  need_venv=1
fi
if [ "$need_venv" = 1 ]; then
  rm -rf .venv
  python3 -m venv .venv --system-site-packages
fi

PY=.venv/bin/python3
PIP=.venv/bin/pip

# apt package python3-lgpio lives in system site-packages
SYS_LIB=$(/usr/bin/python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null || true)
if [ -n "$SYS_LIB" ]; then
  export PYTHONPATH="${SYS_LIB}${PYTHONPATH:+:$PYTHONPATH}"
fi

if ! "$PY" -c "import numpy, cv2, gpiozero" 2>/dev/null; then
  "$PIP" install --upgrade pip
  "$PIP" install -r requirements.txt
fi
if ! "$PY" -c "import lgpio" 2>/dev/null; then
  echo "Missing lgpio — run:  sudo apt install -y python3-lgpio  &&  bash scripts/install_pi.sh" >&2
  exit 1
fi

export GPIOZERO_PIN_FACTORY=lgpio
export PYTHONUNBUFFERED=1
exec "$PY" -u main.py

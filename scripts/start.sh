#!/bin/bash
# Wait for camera and GPIO after power-on, then start the car.
set -e
cd /home/pi/smart-car-pi
sleep 8
if [ -f .venv/bin/python3 ]; then
  PY=.venv/bin/python3
else
  PY=python3
fi
export GPIOZERO_PIN_FACTORY=lgpio
export PYTHONUNBUFFERED=1
exec "$PY" -u main.py

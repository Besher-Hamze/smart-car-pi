#!/bin/bash
# One-time setup on Raspberry Pi. Run: bash scripts/install_pi.sh
set -e
cd "$(dirname "$0")/.."

sudo systemctl stop smart-car 2>/dev/null || true

sudo apt update
sudo apt install -y python3-venv python3-full python3-lgpio python3-numpy

rm -rf .venv
python3 -m venv .venv --system-site-packages
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if ! .venv/bin/python3 -c "import lgpio, cv2, numpy, gpiozero"; then
  echo "Setup failed — check errors above." >&2
  exit 1
fi

echo "OK — sudo systemctl start smart-car"

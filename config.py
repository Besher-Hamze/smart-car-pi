# Pins, speed, and colors. Change only this file if something is inverted.

# L298N (BCM GPIO)
PIN_LEFT_PWM = 12
PIN_LEFT_IN1 = 17
PIN_LEFT_IN2 = 27
PIN_RIGHT_PWM = 13
PIN_RIGHT_IN3 = 22
PIN_RIGHT_IN4 = 23

# HC-SR04. ECHO through a 1k + 2k divider to 3.3V. Never 5V into the Pi.
PIN_TRIG = 5
PIN_ECHO = 6
# Stop when something is this close (cm). Resume after CLEAR_CM.
STOP_CM = 18
CLEAR_CM = 26

# If a side runs backward, set it True.
INVERT_LEFT = False
INVERT_RIGHT = False

# Slow on the straight. Turns stay sharp (do not cut SPEED while steering).
SPEED = 32
TURN = 90
MAX_SPEED = 100
SLOW_IN_TURN = 0.45
STEER_GAIN = 0.72
# Bend ahead (keep low — stops zig-zag).
LOOK_GAIN = 0.62
HEAD_GAIN = 0.75
LANE_MARGIN = 0.26
AUTO_THR_MIN = 0.62
AUTO_PWM_FLOOR = 0.48
# Max steering change per frame (smooth).
STEER_STEP = 0.17
STEER_DEAD = 0.025
AHEAD_GAIN = 0.48
# Where to predict path in bird view (0=top/far, 1=bottom/near). Higher = turn later.
AHEAD_Y = 0.40
# Ignore bird rows above this (checkerboard / poster at horizon).
BIRD_SCAN_TOP = 0.26
AHEAD_ROWS_MIN = 0.32
# Row must be mostly road surface between yellow edges (gray or white dash).
ASPHALT_ROW_MIN = 0.18
WHITE_ROW_MIN = 3
WHITE_BLEND = 0.58
BACKUP_SPEED = 38
LOST_REVERSE = 40
# Auto: drive a short burst, stop, look, decide again.
STEP_SEC = 0.50
PAUSE_SEC = 0.15

# Laptop / phone buttons while MANUAL is selected.
MANUAL_SPEED = 40
MANUAL_TURN = 80
# Speed slider on the page (percent PWM). You change this live.
SPEED_MIN = 15
SPEED_MAX = 90
# If the browser stops sending keys, stop the car.
HOLD_TIMEOUT = 1.8

# TRAIN button: how many epochs (you set this on the page).
TRAIN_EPOCHS = 40
TRAIN_EPOCHS_MIN = 1
TRAIN_EPOCHS_MAX = 200

# Camera + stream (smaller = faster capture + MJPEG).
WIDTH = 424
HEIGHT = 240
CAM_FPS = 30
# If yellow still looks cyan/blue on the page, set this True (or False) and restart.
CAMERA_SWAP_RB = True
# Floor strip: include the road ahead, not just the bumper.
ROI_TOP = 0.28

# Yellow hug (camera only): keep yellow line near this X (0=left, 1=right).
FOLLOW_W = 200
FOLLOW_H = 128
HUG_Y_MIN = 0.20
HUG_MIN_AREA = 24
# 1 = follow ONE yellow line only (even if two are visible). 2 = center between both.
YELLOW_LINES = 1
# Which line when two show up: "left" | "right" | "inner" | "outer"
YELLOW_LINE_SIDE = "right"
# Where that line should sit in the image (0=left edge, 1=right). Right line ~0.64, left ~0.36.
YELLOW_TARGET_X = 0.64
YELLOW_TARGET_LEFT = 0.36
HUG_GAIN = 1.08
HUG_SEARCH = 0.58

# White dashed center line: printed dashes are pale, not lamp-white.
WHITE_LO = (0, 0, 125)
WHITE_HI = (179, 70, 255)
# Yellow frame: keep working when the paint is dim.
YELLOW_LO = (10, 18, 35)
YELLOW_HI = (45, 255, 255)
CYAN_LO = (78, 25, 70)
CYAN_HI = (112, 255, 255)
# Asphalt: gray or bluish speckled mat (not grass, not black void).
ASPHALT_S_MAX = 82
ASPHALT_V_MIN = 34
ASPHALT_V_MAX = 190
ROAD_HUE_LO = 68
ROAD_HUE_HI = 135
ROAD_S_MAX = 82
# Lamp glare on the mat — not a dash line.
GLARE_V_MIN = 208
GLARE_S_MAX = 48
# Matte black surround + inner void — not drivable.
BLACK_V_MAX = 58
BLACK_S_MAX = 95
AUTO_SPEED_DEFAULT = 32

# If the car steers away from the line, set this to -1.
STEER_SIGN = 1

# AUTO: "pilot" = REC/TRAIN model. "follow" = yellow line hug.
AUTO_DRIVE = "follow"
# Floor on learned throttle (0..1). Raise if the car crawls in AUTO.
PILOT_THR_MIN = 0.42
PILOT_STEER_STEP = 0.14

STREAM_PORT = 8080
STREAM_JPEG_QUALITY = 72

# Donkeycar camera size (width x height of the net input).
PILOT_W = 160
PILOT_H = 120

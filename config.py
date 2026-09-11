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
TURN = 80
MAX_SPEED = 100
SLOW_IN_TURN = 0.0
STEER_GAIN = 1.05
# Far scan (road ahead) and bend. Higher = earlier / harder turns.
LOOK_GAIN = 1.25
HEAD_GAIN = 0.75
# Reverse to find the printed road again.
BACKUP_SPEED = 38
LOST_REVERSE = 8
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

# Enough pixels to see dashes; HSV on the bottom ROI is still cheap.
WIDTH = 640
HEIGHT = 480
CAM_FPS = 20
# Floor strip: far row is higher so a bend is seen before the wheels.
ROI_TOP = 0.62

# White dashed line only: bright + not colorful.
WHITE_LO = (0, 0, 185)
WHITE_HI = (179, 50, 255)

# If the car steers away from the line, set this to -1.
STEER_SIGN = 1

STREAM_PORT = 8080

# Donkeycar camera size (width x height of the net input).
PILOT_W = 160
PILOT_H = 120

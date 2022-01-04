import os

""" Paths Configurations """

# path to save the captured images into
OUTPUT_IMAGES_PATH = "exp/"
os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)


""" General Configurations """
IMAGE_SIZE = (1080, 720)

# maximum number of faces to be detected
MAX_NUMBER_FACES = 1

# whether to draw visualizations or not
VIS = False

""" Thresholds Configurations """

# set threshold for pitch index
# which means the value of pitch index between [-PITCH_THRESHOLD, +PITCH_THRESHOLD] will be considered
# the value of PITCH_THRESHOLD will always be between [0,1] inclusive
PITCH_THRESHOLD = 0.2
assert 0 <= PITCH_THRESHOLD <= 1, "the value of PITCH_THRESHOLD will always be between [0,1] inclusive"

# set threshold for yaw index
# which means the value of yaw index between [-YAW_THRESHOLD, +YAW_THRESHOLD] will be considered
# the value of YAW_THRESHOLD will always be between [0,1] inclusive
YAW_THRESHOLD = 0.2
assert 0 <= YAW_THRESHOLD <= 1, "the value of YAW_THRESHOLD will always be between [0,1] inclusive"

# set threshold for roll angle
# which means the value of rol angle between [-ROLL_THRESHOLD, +ROLL_THRESHOLD] will be considered
# the value of ROLL_THRESHOLD will always be between [0,90] inclusive
ROLL_THRESHOLD = 20
assert 0 <= ROLL_THRESHOLD <= 90, "the value of ROLL_THRESHOLD will always be between [0,90] inclusive"

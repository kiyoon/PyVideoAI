import os

DATA_DIR = "../data"
DEFAULT_EXPERIMENT_ROOT = os.path.join(DATA_DIR, "experiments")
KEY_INI_PATH = "../tools/key.ini"

###########################
### DO NOT MODIFY BELOW ###
# if relative path, turn into absolute
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
def turn_absolute(path):
    if not os.path.isabs(path):
        return os.path.realpath(os.path.join(_SCRIPT_DIR, path))
    else:
        return path

DATA_DIR = turn_absolute(DATA_DIR)
DEFAULT_EXPERIMENT_ROOT = turn_absolute(DEFAULT_EXPERIMENT_ROOT)
KEY_INI_PATH = turn_absolute(KEY_INI_PATH)


PYVIDEOAI_DIR = turn_absolute("..")     # root PyVideoAI directory

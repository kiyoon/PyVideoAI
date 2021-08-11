import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../densesample_TC_crop224_8x8_largejit_plateau_3scrop10tcrop.py').read())

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../sparsesample_TC_crop224_8frame_largejit_plateau_10scrop.py').read())

input_frame_length = 9
ordering = get_orderedTC_ordering(input_frame_length)   # when you change input frame number, the ordering has to be recalculated.

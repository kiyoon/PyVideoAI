import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py').read())  # NOTE: use RGB version, and change sample_index_code to `TDN_GreyST`

greyscale=True
sample_index_code = 'TDN_GreyST'
clip_grad_max_norm = 20

import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
exec(open(f'{_SCRIPT_DIR}/../sparsesample_GreyST_crop224_8frame_largejit_plateau_10scrop.py').read())

sample_index_code = 'TDN'
clip_grad_max_norm = 20

_exec_relative_('../sparsesample_onehot_crop224_8frame_largejit_plateau.py')
_exec_relative_('../tsm_resnet50_nopartialbn_base.py')

base_learning_rate = 5e-6

pretrained = None

pretrained_path = '/home/kiyoon/fast/experiments_labelsmooth/epic55_verb/tsm_resnet50_nopartialbn/onehot/version_002/weights/best.pth'
def load_pretrained(model):
    loader.model_load_weights_GPU_partial_nostrict(model, pretrained_path)

import torch
from torch import optim

from pyvideoai.models.TRN.models import TSN

def load_model(num_classes, input_frame_length):
    #class_counts = (num_classes,352)
    class_counts = num_classes
    segment_count = input_frame_length
    base_model = 'BNInception'

    # By default it is pretrained
    model = TSN(class_counts, segment_count, 'RGB',
            base_model = base_model,
            consensus_type = 'TRNmultiscale',
            dropout = 0.8,
            img_feature_dim=256,
            partial_bn=True)
            
    return model

def NCTHW_to_model_input_shape(inputs):
    N, C, T, H, W = inputs.shape
    # N, C, T, H, W -> N, T, C, H, W -> N, TC, H, W
    return inputs.permute(0,2,1,3,4).reshape((N, -1, H, W))

def model_input_shape_to_NTHWC(inputs, num_channels=3):
    N, TC, H, W = inputs.shape
    # N, TC, H, W -> N, T, C, H, W -> N, T, H, W, C
    return inputs.reshape((N, -1, num_channels, H, W)).permute(0,1,3,4,2)



def get_optim_policies(model):
    # return model.parameters()     # no policies
    return model.get_optim_policies()


ddp_find_unused_parameters = True 
use_amp = True


# input configs
input_normalise = False 
input_bgr = True
input_mean = [104, 117, 128]
input_std = [1]


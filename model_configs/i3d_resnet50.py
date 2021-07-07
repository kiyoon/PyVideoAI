from pyvideoai.models.slowfast.video_model_builder import ResNetModel
from torch import optim

import torch.distributed as dist
from pyvideoai.utils import loader

def load_model(num_classes, input_frame_length, crop_size=224, input_channel_num=[3]):
    model = ResNetModel('i3d', 50, num_classes, input_frame_length, crop_size=crop_size, input_channel_num=input_channel_num)
    return model


from pyvideoai.slowfast.utils import checkpoint as cu
from pyvideoai.config import DATA_DIR
import os
kinetics400_pretrained_path_32x2 = os.path.join(DATA_DIR, 'pretrained', 'kinetics400/i3d_resnet50/i3d_baseline_32x2_IN_pretrain_400k.pkl')
kinetics400_pretrained_path_8x8 = os.path.join(DATA_DIR, 'pretrained', 'kinetics400/i3d_resnet50/I3D_8x8_R50.pkl')
def load_pretrained_kinetics400(model, pretrained_path=kinetics400_pretrained_path_32x2):
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    cu.load_checkpoint(
        pretrained_path,
        model,
        world_size > 1,
        inflation=False,
        convert_from_caffe2=True)

hmdb_pretrained_path_8x8 = os.path.join(DATA_DIR, 'pretrained', 'hmdb/i3d_resnet50/crop224_lr0001_batch8_8x8_largejit_plateau_1scrop5tcrop_split1-epoch_0199.pth')
def load_pretrained_hmdb(model, pretrained_path=hmdb_pretrained_path_8x8):
    loader.model_load_weights_GPU(model, pretrained_path)

def model_input_shape_to_NTHWC(inputs):
    # [inputs] -> inputs
    # N, C, T, H, W -> N, T, H, W, C
    return inputs[0].permute((0,2,3,4,1))

def NCTHW_to_model_input_shape(inputs):
    return [inputs]


ddp_find_unused_parameters = False
#use_amp = True

# input configs
input_normalise = False
input_bgr = False
input_mean = [114.75, 114.75, 114.75]
input_std = [57.375, 57.375, 57.375]


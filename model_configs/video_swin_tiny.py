"""
Video Swin Transformer Tiny.
The parameters are taken from the two files:
https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/_base_/models/swin/swin_tiny.py
https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py

It will load the Kinetics-400 pre-trained model smoothly.
"""
import os
import torch
from torch.nn import Module
from pyvideoai.models.video_swin import SwinTransformer3DWithHead
from pyvideoai.config import DATA_DIR
from pyvideoai.utils.loader import model_load_state_dict_nostrict

import logging
logger = logging.getLogger(__name__)

def load_model(num_classes, input_channel_num=3, input_frame_length=None, crop_size=None):
    if input_frame_length is not None:
        logger.warning('This model does not require input_frame_length. It will give you the same output shape regardless of the length.')
    if crop_size is not None:
        logger.warning('This model does not require crop_size. It will give you the same output shape regardless of the length.')

    model = SwinTransformer3DWithHead(num_classes = num_classes,
                patch_size=(2,4,4),
                in_chans=input_channel_num,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=(8,7,7),
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                patch_norm=True,
                head_in_channels=768,
                head_spatial_type='avg',
                head_dropout_ratio=0.5,
                )

    return model


kinetics400_pretrained_path_32x2 = os.path.join(DATA_DIR, 'pretrained', 'kinetics400/video_swin_tiny/swin_tiny_patch244_window877_kinetics400_1k.pth')
def load_pretrained_kinetics400(model, pretrained_path=kinetics400_pretrained_path_32x2):
    checkpoint = torch.load(pretrained_path)
    model_load_state_dict_nostrict(model, checkpoint['state_dict'])


def get_optim_policies(model):
    return model.parameters()     # no policies


def model_input_shape_to_NTHWC(inputs):
    # N, C, T, H, W -> N, T, H, W, C
    return inputs.permute((0,2,3,4,1))

def NCTHW_to_model_input_shape(inputs):
    return inputs


# If you need to extract features, use this. It can be defined in exp_configs too.
def feature_extract_model(model, featuremodel_name):
    if featuremodel_name == 'features':
        class FeatureExtractModel(Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    self.is_ddp = True
                else:
                    self.is_ddp = False

            def forward(self, x):
                #batch_size = x.shape[0]
                if self.is_ddp:
                    backbone_features = self.model.module.features(x)
                else:
                    backbone_features = self.model.features(x)
                # It considers frames are image batch. Disentangle so you get actual video batch and number of frames.
                # Average over frames, width and height.
                # (N, C, T, H, W)
                # Use average pool 3D?
                return torch.mean(torch.mean(torch.mean(backbone_features, dim=-1), dim=-1), dim=-1)


        return FeatureExtractModel(model)

    elif featuremodel_name == 'logits':
        return model
    else:
        raise ValueError(f'Unknown feature model: {featuremodel_name}')


ddp_find_unused_parameters = False
use_amp = True

# input configs
input_normalise = False
input_bgr = False
input_mean = [123.675, 116.28, 103.53]
input_std = [58.395, 57.12, 57.375]

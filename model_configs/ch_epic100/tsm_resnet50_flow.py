import torch
from torch.nn import Module
from torch import optim

from pyvideoai.models.epic.tsm import TSM
from pyvideoai.config import PYVIDEOAI_DIR
from pyvideoai.utils.loader import model_load_state_dict_nostrict
import os

from video_datasets_api.epic_kitchens_100.definitions import NUM_VERB_CLASSES, NUM_NOUN_CLASSES

pretrained_path = os.path.join(PYVIDEOAI_DIR, 'data', 'pretrained', 'epic100', 'tsm_flow.ckpt')

def load_model(num_classes = NUM_VERB_CLASSES, input_frame_length = 8):
    """
    num_classes can be integer or tuple
    """

    assert 0 < num_classes <= NUM_VERB_CLASSES+NUM_NOUN_CLASSES

    #class_counts = (num_classes,352)
    class_counts = NUM_VERB_CLASSES+NUM_NOUN_CLASSES
    segment_count = input_frame_length
    base_model = 'resnet50'
    pretrained = None

    model = TSM(class_counts, segment_count, 'Flow',
            base_model = base_model,
            pretrained=pretrained, dropout=0.7)

    device = torch.cuda.current_device()
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[6:] # remove `model.`
        new_state_dict[name] = v

    model_load_state_dict_nostrict(model, new_state_dict, partial=True)

    del checkpoint, new_state_dict
    class VerbModel(Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)[:num_classes]

    return VerbModel(model)


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
    return model.model.get_optim_policies()

# If you need to extract features, use this. It can be defined in exp_configs too.
def feature_extract_model(model):
    class FeatureExtractModel(Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model 
        def forward(self, x):
            batch_size = x.shape[0]
            imagenet_features = self.model.features(x)
            # It considers frames are image batch. Disentangle so you get actual video batch and number of frames.
            return imagenet_features.view(batch_size, imagenet_features.shape[0] // batch_size, *imagenet_features.shape[1:])

    return FeatureExtractModel(model)

ddp_find_unused_parameters = True
use_amp = True


# input configs
input_normalise = True
input_bgr = False
input_mean = [0.5]
input_std = [0.226]


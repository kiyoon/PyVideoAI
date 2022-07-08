import torch
from torch.nn import Module
from torch import optim

from pyvideoai.models.epic.tsm import TSM

def load_model(num_classes, input_frame_length, pretrained='imagenet'):
    class_counts = num_classes
    segment_count = input_frame_length
    base_model = 'resnet50'

    model = TSM(class_counts, segment_count, 'RGB',
            base_model = base_model,
            consensus_type='avg',
            partial_bn = False,
            dropout=0.5,
            pretrained=pretrained)


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
                batch_size = x.shape[0]
                if self.is_ddp:
                    imagenet_features = self.model.module.features(x)
                else:
                    imagenet_features = self.model.features(x)
                # It considers frames are image batch. Disentangle so you get actual video batch and number of frames.
                # Average over frames
                #return torch.mean(imagenet_features.view(batch_size, imagenet_features.shape[0] // batch_size, *imagenet_features.shape[1:]), dim=1)
                return imagenet_features.view(batch_size, imagenet_features.shape[0] // batch_size, *imagenet_features.shape[1:])


        return FeatureExtractModel(model)

    elif featuremodel_name == 'logits':
        return model
    else:
        raise ValueError(f'Unknown feature model: {featuremodel_name}')


ddp_find_unused_parameters = False
use_amp = True


# input configs
input_normalise = True
input_bgr = False
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

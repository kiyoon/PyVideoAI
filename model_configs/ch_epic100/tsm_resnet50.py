import torch
from torch.nn import Module
from torch import optim

from pyvideoai.models.epic.tsm import TSM
from pyvideoai.config import PYVIDEOAI_DIR
from pyvideoai.utils.loader import model_load_state_dict_nostrict
import os

from video_datasets_api.epic_kitchens_100.definitions import NUM_VERB_CLASSES, NUM_NOUN_CLASSES

pretrained_path = os.path.join(PYVIDEOAI_DIR, 'data', 'pretrained', 'epic100', 'tsm_rgb.ckpt')

def load_model(num_classes = NUM_VERB_CLASSES, input_frame_length = 8, pretrained = 'epic100'):
    """
    num_classes can be integer or tuple
    """

    assert 0 < num_classes <= NUM_VERB_CLASSES+NUM_NOUN_CLASSES
    assert pretrained in ['epic100', 'imagenet', 'random']

    #class_counts = (num_classes,352)
    class_counts = NUM_VERB_CLASSES+NUM_NOUN_CLASSES
    segment_count = input_frame_length
    base_model = 'resnet50'

    if pretrained == 'imagenet':
        basemodel_pretrained = 'imagenet'
    else:
        basemodel_pretrained = None

    model = TSM(class_counts, segment_count, 'RGB',
            base_model = base_model,
            pretrained=basemodel_pretrained, dropout=0.7)


    if pretrained == 'epic100':
        device = torch.cuda.current_device()
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[6:] # remove `model.`
            new_state_dict[name] = v

        model_load_state_dict_nostrict(model, new_state_dict, partial=True)

        del checkpoint, new_state_dict

    if num_classes != NUM_VERB_CLASSES + NUM_NOUN_CLASSES:
        # Only change the classifier if it is different from the pre-trained model.
        model.new_fc = torch.nn.Linear(2048, num_classes)
    return model
#    class VerbModel(Module):
#        def __init__(self, model):
#            super().__init__()
#            self.model = model
#        def forward(self, x):
#            return self.model(x)[:num_classes]
#
#    return VerbModel(model)


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



def freeze_base_model(model):
    # requires_grad has to be set before DDP.
    # Destroy the DDP instance and create new one.
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_normal = model.module
        for param in model_normal.base_model.parameters():
            param.requires_grad = False

        # load settings from the previous DDP model
        model = nn.parallel.DistributedDataParallel(
            module=model_normal, device_ids=model.device_ids, output_device=model.output_device,
            find_unused_parameters=model.find_unused_parameters
        )
    else:
        for param in model.base_model.parameters():
            param.requires_grad = False
    return model

def initialise_classifier(model):
    # DDP model has to be in sync over the processes.
    # Just in case, convert it to normal model and then reconstruct the DDP.
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_normal = model.module
        model_normal._initialise_layer(model_normal.new_fc)
        # load settings from the previous DDP model
        model = nn.parallel.DistributedDataParallel(
            module=model_normal, device_ids=model.device_ids, output_device=model.output_device,
            find_unused_parameters=model.find_unused_parameters
        )
    else:
        model._initialise_layer(model.new_fc)

    return model



ddp_find_unused_parameters = True
use_amp = True


# input configs
input_normalise = True
input_bgr = False
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

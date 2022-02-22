import torch

from pyvideoai.models.feature_model import FeatureModel

def load_model(num_classes, input_feature_dim, num_layers=2, num_units=128):
    model = FeatureModel(input_feature_dim, num_classes=num_classes, num_layers=num_layers, num_units=num_units)

    return model




def get_optim_policies(model):
    return model.parameters()     # no policies


ddp_find_unused_parameters = False
use_amp = True

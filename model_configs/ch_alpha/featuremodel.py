import torch
from torch import optim

from ..models.epic.feature_model import feature_model

def load_model(num_classes, input_frame_length):
    #class_counts = (num_classes,352)
    class_counts = num_classes
    segment_count = input_frame_length

    model = feature_model(input_frame_length*10*7, num_classes = num_classes)
    return model


def load_pretrained(model):
    return None

def get_optim_policies(model):
    return model.parameters()     # no policies


ddp_find_unused_parameters = False

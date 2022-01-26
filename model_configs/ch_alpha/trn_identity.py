import torch
from torch import optim

from ..models.epic.tsn import STRN

def load_model(num_classes, input_frame_length):
    #class_counts = (num_classes,352)
    class_counts = num_classes
    segment_count = input_frame_length
    base_model = 'identity'
#    repo = 'epic-kitchens/action-models'
#    model = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
#            base_model = base_model,
#            pretrained='epic-kitchens')

    # Use TRN if you want to add an fc layer after the base model.
    # Use STRN if you don't.
    model = STRN(class_counts, segment_count, 'RGB',
            base_model = base_model,
            img_feature_dim=70)


    return model


def load_pretrained(model):
    return None

def get_optim_policies(model):
    return model.parameters()     # no policies

ddp_find_unused_parameters = True

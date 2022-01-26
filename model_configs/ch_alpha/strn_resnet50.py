import torch
from torch import optim

from ..models.epic.tsn import STRN, TRN

def load_model(num_classes, input_frame_length):
    #class_counts = (num_classes,352)
    class_counts = num_classes
    segment_count = input_frame_length
    base_model = 'resnet50rn'
    pretrained = 'imagenet'
#    repo = 'epic-kitchens/action-models'
#    model = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
#            base_model = base_model,
#            pretrained='epic-kitchens')

    # Use TRN if you want to add an fc layer after the base model.
    # Use STRN if you don't.
    model = STRN(class_counts, segment_count, 'RGB',
            base_model = base_model,
            pretrained=pretrained)


    return model


def load_pretrained(model):
    return None

def model_predict_NCTHW(model, inputs):
    N, C, T, H, W = inputs.shape
    
    return model(inputs.permute(0,2,1,3,4).reshape((N, -1, H, W)))

def get_optim_policies(model):
    # return model.parameters()     # no policies
    return model.get_optim_policies()

dataloader_type = 'sparse_frames'   # video_clip, sparse_video, sparse_frames

ddp_find_unused_parameters = True


# input configs
input_normalise = True
input_bgr = False
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]


import torch
from torch import optim

from ..models.epic.tsn import VideoLSTM

def load_model(num_classes, input_frame_length):
    #class_counts = (num_classes,352)
    class_counts = num_classes
    segment_count = input_frame_length
    base_model = 'resnet50'
    #pretrained = 'epic_kitchens'
    pretrained = 'imagenet'
#    repo = 'epic-kitchens/action-models'
#    model = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
#            base_model = base_model,
#            pretrained='epic-kitchens')

    model = VideoLSTM(class_counts, segment_count, 'RGB',
            base_model = base_model,
            pretrained=pretrained)


    return model


def load_pretrained(model):
    return None


def get_optim_policies(model):
    # return model.parameters()     # no policies
    return model.get_optim_policies()


dataloader_type = 'sparse_frames'   # video_clip, sparse_video, sparse_frames
#dataloader_type = 'sparse_video'   # video_clip, sparse_video, sparse_frames

ddp_find_unused_parameters = True


# input configs
input_normalise = True
input_bgr = False
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]


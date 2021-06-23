from torch import nn
from torchvision import models
from video_datasets_api.imagenet.definitions import NUM_CLASSES as num_classes_imagenet

def load_model(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    if num_classes != num_classes_imagenet:
        # replace the classifier
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def model_input_shape_to_NHWC(inputs):
    # N, C, H, W -> N, H, W, C
    return inputs.permute(0,2,3,1)

def NCHW_to_model_input_shape(inputs):
    return inputs


ddp_find_unused_parameters = False

# input configs
input_normalise = True
input_bgr = False
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]


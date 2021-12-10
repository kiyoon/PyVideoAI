import torch
from torch import optim, nn

from pyvideoai.models.epic.tsm import TSM
import pyvideoai.models.timecycle.videos.model_test as video3d

import logging
logger = logging.getLogger(__name__)

def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

import os
from pyvideoai.config import PYVIDEOAI_DIR
corr_model_checkpoint_path = os.path.join(PYVIDEOAI_DIR, 'tools', 'timecycle', 'checkpoint_14.pth.tar')


class CorrModel(nn.Module):
    def __init__(self, action_model, corr_model = None, corr_model_checkpoint_path = 'checkpoint_14.pth.tar', num_channels = 2):
        super(CorrModel, self).__init__()
        if corr_model is None:
            self.corr_model = video3d.CycleTime(trans_param_num=3)

            assert os.path.isfile(corr_model_checkpoint_path), f'Error: no checkpoint file found! {corr_model_checkpoint_path}'
            device = torch.cuda.current_device()
            checkpoint = torch.load(corr_model_checkpoint_path, map_location=f'cuda:{device}')
            #start_epoch = checkpoint['epoch']
            partial_load(checkpoint['state_dict'], self.corr_model)
            del checkpoint
        else:
            self.corr_model = corr_model

        assert num_channels in [2, 3], "2 for flow x y, 3 for flow x y and its confidence."
        self.num_channels = num_channels

        # freeze corr model params
        for param in self.corr_model.parameters():
            param.requires_grad = False

        self.action_model = self._convert_action_model_to_action_corr_model(action_model)

    def _convert_action_model_to_action_corr_model(self, action_model):
        """
        Action model (ResNet base) + corr model
        Add corr feature after layer2
        """
        # Make convolutional block for correlation feature processing
        from torchvision.models.resnet import BasicBlock, Bottleneck
        action_model.base_model.corr_conv_layers = nn.Sequential(
                nn.Conv2d(self.num_channels, 256, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))

        action_model.base_model.after_layer2_match_channel = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))

        #action_model.base_model.layer2 = action_model.base_model._make_layer(block, 128, 2, stride=2, dilate=False)   # instead of 4 blocks, 2 blocks. Output = 256 channels hopefully

        action_model.base_model.corr_model = self.corr_model
        action_model.base_model.num_segments = action_model.num_segments
        action_model.base_model.corr_num_channels = self.num_channels
        action_model.base_model.device = torch.cuda.current_device()

        # Patching original forward function to something else.
        from torch import Tensor
        from types import MethodType
        def features(self, x: Tensor) -> Tensor:
            # Get corresponence from RGB input
            # Assume NT C H W input
            if x.dim() == 4:
                NT, C, H, W = x.shape
                T = self.num_segments
                N = NT // T
                video = x.view(N, T, C, H, W)
            else:
                raise ValueError(f'x.shape {x.shape} not recognised.')

            imgs_tensor = video.view(N*T, 1, C, H, W)
            target_frames = list(range(1,T)) + [T-1]    # repeat last frame
            target_tensor = video[:,target_frames,...].reshape(N*T, 1, C, H, W)

            self.corr_model.eval()
            corr_features = self.corr_model(imgs_tensor, target_tensor) # (N * (T), W/8*H/8, H/8, W/8)

            # make flows from the corr feature by finding max activation
            corr_features_orig = corr_features.view(N, T, W//8*H//8, H//8*W//8)
            if self.corr_num_channels == 2:
                corr_features_argmax = torch.argmax(corr_features_orig, dim=3) # (N, T, W/8*H/8)
            elif self.corr_num_channels == 3:
                corr_features_confidence, corr_features_argmax = corr_features_orig.max(dim=3)
                corr_features_confidence = corr_features_confidence.view(N, T, W//8, H//8, 1)

            corr_features = corr_features_argmax.view(N, T, W//8, H//8, 1).repeat(1,1,1,1,2) # (N, T, W/8, H/8, 2). The last dim for flow x and y

            # Convert idx to x and y
            corr_features[..., 0] %= W//8   # flow dest x (absolute point)
            corr_features[..., 1] = torch.div(corr_features[..., 1], W//8, rounding_mode='floor')   # flow dest y (absolute point)

            # We have to make flow relative.
            # Subtract indices

            X = torch.arange(0, W//8, device=self.device).view(1,1,W//8,1)
            corr_features[..., 0] -= X
            Y = torch.arange(0, H//8, device=self.device).view(1,1,1,H//8)
            corr_features[..., 1] -= Y

            # Type conversion to float, and scale between -1 and 1
            corr_features = corr_features.float()
            corr_features[..., 0] /= W//8//2
            corr_features[..., 1] /= H//8//2

            # 3rd channel which is confidence score
            if self.corr_num_channels == 3:
                corr_features = torch.cat([corr_features, corr_features_confidence], dim=-1)

            # view
            corr_features = corr_features.permute(0,1,4,3,2) # (N, T, C=2 or 3, H//8, W//8)
            corr_features = corr_features.view(N*T, self.corr_num_channels, H//8, W//8)

            # Original Resnet feature function
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)

            # correlation concat
            # x.shape NT, 512, 28, 28
            x = self.after_layer2_match_channel(x)          # (NT, 256, 28, 28)
            import ipdb; ipdb.set_trace()
            corr_features_conv = self.corr_conv_layers(corr_features)    # (NT, 256, 28, 28)
            x = torch.cat((x, corr_features_conv), dim=1)   # (NT, 512, 28, 28)
            

            x = self.layer3(x)
            x = self.layer4(x)

            return x

        # Replacing function requires MethodType. It's a hacky patch, not recommended, but it's the easiest to implement for now.
        action_model.base_model.features = MethodType(features, action_model.base_model)
        
        return action_model


    def forward(self, x):
        return self.action_model(x)

    def get_optim_policies(self):
        return self.flow_model.get_optim_policies()

def load_model(num_classes, input_frame_length, num_channels=2):
    class_counts = num_classes
    segment_count = input_frame_length
    base_model = 'resnet50'
    pretrained = 'imagenet'

    RGB_model = TSM(class_counts, segment_count, 'RGB',
            base_model = base_model,
            #new_length = segment_count-1,
            new_length = 1,
            partial_bn = False,
            dropout = 0.5,
            pretrained = pretrained 
            )


    corr_model = CorrModel(RGB_model, corr_model_checkpoint_path = corr_model_checkpoint_path, num_channels=num_channels)

    return corr_model


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


ddp_find_unused_parameters = False
use_amp = True


# input configs
input_normalise = True
input_bgr = False
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]


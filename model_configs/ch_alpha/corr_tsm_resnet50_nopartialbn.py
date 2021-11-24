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
    def __init__(self, flow_model, corr_model = None, corr_model_checkpoint_path = 'checkpoint_14.pth.tar'):
        super(CorrModel, self).__init__()
        self.flow_model = flow_model
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

        # freeze corr model params
        for param in self.corr_model.parameters():
            param.requires_grad = False

        self.device = torch.cuda.current_device()


    def forward(self, x):
        if x.dim() == 4:
            N, TC, H, W = x.shape
            T = TC // 3
            C = 3
            x = x.view(N, T, C, H, W)
        elif x.dim() == 5:
            N, T, C, H, W = x.shape
        else:
            raise ValueError(f'x.shape {x.shape} not recognised.')

        imgs_tensor = x[:,0:T-1,...].contiguous()   # N, T-1, C, H, W
        target_tensor = x[:,T-1:T,...].contiguous() # N, 1, C, H, W

        self.corr_model.eval()
        corr_features = self.corr_model(imgs_tensor, target_tensor) # (N * (T-1), W/8*H/8, H/8, W/8)

        # make flows from the corr feature by finding max activation
        corr_features = corr_features.view(N, T-1, W//8*H//8, H//8*W//8)
        corr_features = torch.argmax(corr_features, dim=3) # (N, T-1, W/8*H/8)
        corr_features = corr_features.view(N, T-1, W//8, H//8, 1).repeat(1,1,1,1,2) # (N, T-1, W/8, H/8, 2). The last dim for flow x and y

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

        # interpolate
        corr_features = corr_features.permute(0,1,4,2,3) # (N, T-1, C=2, W//8, H//8)
        corr_features = corr_features.reshape(N, (T-1)*2, W//8, H//8)
        corr_features = nn.functional.interpolate(corr_features, scale_factor = 8, mode = 'bilinear', align_corners=False)   # (N, (T-1)*2 , W, H)

        flow_model_output = self.flow_model(corr_features)
        logger.info(f'{corr_features.shape = }')
        logger.info(f'{flow_model_output.shape = }')
        return flow_model_output

    def get_optim_policies(self):
        return self.flow_model.get_optim_policies()

class ActionCorrModel(nn.Module):
    def __init__(self, num_classes, num_segments, crop_size, action_model, corr_model = None, corr_model_checkpoint_path = 'checkpoint_14.pth.tar', enable_corr_model=True):
        super(ActionCorrModel, self).__init__()
        self.action_model = action_model
        self.enable_corr_model = enable_corr_model
        if self.enable_corr_model:
            if corr_model is None:
                self.corr_model = video3d.CycleTime(trans_param_num=3)

                assert os.path.isfile(corr_model_checkpoint_path), 'Error: no checkpoint directory found!'
                device = torch.cuda.current_device()
                checkpoint = torch.load(corr_model_checkpoint_path, map_location=f'cuda:{device}')
                #start_epoch = checkpoint['epoch']
                partial_load(checkpoint['state_dict'], self.corr_model)
                del checkpoint

            else:
                self.corr_model = corr_model

            # freeze corr model params
            for param in self.corr_model.parameters():
                param.requires_grad = False


        # classifier
        backbone_feature_dim = 2048
        action_feature_dim = backbone_feature_dim * num_segments
        corr_feature_dim = crop_size ** 2 // 8 // 8 * (num_segments-1)

        if self.enable_corr_model:
            self.output = nn.Linear(action_feature_dim+corr_feature_dim, num_classes)
        else:
            self.output = nn.Linear(action_feature_dim, num_classes)


    def forward(self, x):
        if x.dim() == 4:
            N, TC, H, W = x.shape
            T = TC // 3
            C = 3
            x = x.view(N, T, C, H, W)
        elif x.dim() == 5:
            N, T, C, H, W = x.shape
        else:
            raise ValueError(f'x.shape {x.shape} not recognised.')
        action_features = self.action_model.features(x) # N*T, F
        action_features = action_features.view(N, -1)

        imgs_tensor = x[:,0:T-1,...].contiguous()   # N, T-1, C, H, W
        target_tensor = x[:,T-1:T,...].contiguous() # N, 1, C, H, W

        if self.enable_corr_model:
            self.corr_model.eval()
            corr_features = self.corr_model(imgs_tensor, target_tensor) # (N, W/8*H/8, H/8, W/8)
            # make flows from the corr feature by finding max activation
            corr_features = corr_features.view(N, T-1, W//8*H//8, H//8*W//8)
            corr_features = torch.argmax(corr_features, dim=3) # (N, T-1, W/8*H/8)
            corr_features = corr_features.view(N, -1)
            
            concat_features = torch.cat((action_features, corr_features), dim=1)

            return self.output(concat_features)
        else:
            return self.output(action_features)

    def get_optim_policies(self):
        return self.action_model.get_optim_policies()

def load_model(num_classes, input_frame_length):
    class_counts = num_classes
    segment_count = input_frame_length
    base_model = 'resnet50'
#    pretrained = 'imagenet'
#
#    action_model = TSM(class_counts, segment_count, 'RGB',
#            base_model = base_model,
#            consensus_type='avg',
#            partial_bn = False,
#            dropout=0.5,
#            pretrained=pretrained)
#
#    action_corr_model = ActionCorrModel(class_counts, segment_count, 224, action_model, corr_model_checkpoint_path = corr_model_checkpoint_path, enable_corr_model=False)

    flow_model = TSM(class_counts, segment_count, 'Flow',
            base_model = base_model,
            new_length = segment_count-1,
            partial_bn = False,
            dropout = 0.5,
            pretrained = None
            )
    corr_model = CorrModel(flow_model, corr_model_checkpoint_path = corr_model_checkpoint_path)

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


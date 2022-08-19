"""
Action model with correspondence (fixed params)
"""
import pyvideoai.models.timecycle.videos.model_test as video3d
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import os
from decord import VideoReader
import decord

from pyvideoai.dataloaders import utils
from pyvideoai.models.epic.tsm import TSM

def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

class ARGS:
    pass
args = ARGS()
args.resume = 'checkpoint_14.pth.tar'
args.crop_size = 224


action_model = TSM(174, 3, 'RGB',
        base_model = 'resnet50',
        partial_bn = False,
        pretrained='imagenet')

class action_corr_model(nn.Module):
    def __init__(self, num_classes, num_segments, crop_size, action_model, corr_model = None, corr_model_checkpoint_path = 'checkpoint_14.pth.tar'):
        super(action_corr_model, self).__init__()
        self.action_model = action_model
        if corr_model is None:
            self.corr_model = video3d.CycleTime(trans_param_num=3)

            assert os.path.isfile(corr_model_checkpoint_path), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(corr_model_checkpoint_path)
            #start_epoch = checkpoint['epoch']
            partial_load(checkpoint['state_dict'], self.corr_model)
            del checkpoint

        else:
            self.corr_model = corr_model


        # classifier
        backbone_feature_dim = 2048
        action_feature_dim = backbone_feature_dim * num_segments
        corr_feature_dim = crop_size ** 2 // 8 // 8 * (num_segments-1)
        self.output = nn.Linear(action_feature_dim+corr_feature_dim, num_classes)


        # freeze corr model params
        for param in self.corr_model.parameters():
            param.requires_grad = False



    def forward(self, x):
        N, T, C, H, W = x.shape
        action_features = self.action_model.features(x) # N*T, F
        action_features = action_features.view(N, -1)

        imgs_tensor = x[:,0:T-1,...].contiguous()   # N, T-1, C, H, W
        target_tensor = x[:,T-1:T,...].contiguous() # N, 1, C, H, W
        print(imgs_tensor.shape)
        print(target_tensor.shape)

        self.corr_model.eval()
        corr_features = self.corr_model(imgs_tensor, target_tensor) # (N, W/8*H/8, H/8, W/8)
        # make flows from the corr feature by finding max activation
        corr_features = corr_features.view(N, T-1, W//8*H//8, H//8*W//8)
        corr_features = torch.argmax(corr_features, dim=3) # (N, T-1, W/8*H/8)
        corr_features = corr_features.view(N, -1)
        
        concat_features = torch.cat((action_features, corr_features), dim=1)

        return self.output(concat_features)





test_video = '/disk/scratch2/s1884147/datasets/something-something-v2/videos/11.webm'

def main():
    model = action_corr_model(174, 3, 224, action_model)

    vr = VideoReader(test_video)
    
    decord.bridge.set_bridge('torch')
    print(len(vr))
    video = vr.get_batch([0,1,2])

    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])

    video = utils.tensor_normalize(video, mean, std, True)
    # T, H, W, C -> C, T, H, W
    video = video.permute(3, 0, 1, 2)

    # Perform data augmentation.
    video, scale_factor_width, scale_factor_height, x_offset, y_offset, is_flipped = utils.spatial_sampling_5(
        video,
        spatial_idx=-1,
        min_scale=224,
        max_scale=336,
        crop_size=args.crop_size,
        random_horizontal_flip=False,
    )
    # C, T, H, W -> T, C, H, W
    video = video.permute(1, 0, 2, 3)
    
    videos = video.unsqueeze(0).expand(2, -1,-1,-1,-1).contiguous() # add batch 2
    print(videos.shape)

    print(model(videos).shape)


if __name__ == '__main__':
    main()

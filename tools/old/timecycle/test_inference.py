import pyvideoai.models.timecycle.videos.model_test as video3d
import torch
import torch.backends.cudnn as cudnn
import os
from decord import VideoReader
import decord

from pyvideoai.dataloaders import utils

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


test_video = '/disk/scratch2/s1884147/datasets/something-something-v2/videos/11.webm'

def main():
    model = video3d.CycleTime(trans_param_num=3)
    model = torch.nn.DataParallel(model).cuda()

    #cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    title = 'videonet'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        partial_load(checkpoint['state_dict'], model)

        del checkpoint

    model.eval()
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
    imgs_tensor = torch.Tensor(1, 1, 3, args.crop_size, args.crop_size)
    target_tensor = torch.Tensor(1, 1, 3, args.crop_size, args.crop_size)

    imgs_tensor[0,0] = video[0]
    target_tensor[0,0] = video[1]

    corrfeat = model(imgs_tensor, target_tensor)
    corrfeat = corrfeat.view(1, corrfeat.size(1), corrfeat.size(2), corrfeat.size(3))
    print(corrfeat.shape)

if __name__ == '__main__':
    main()

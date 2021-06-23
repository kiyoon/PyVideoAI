import torch
import numpy as np
from PIL import Image

from dataloaders.frames_sparsesample_dataset import FramesSparsesampleDataset

from dataloaders.trn.transforms import *
from dataloaders.trn.datasets_video import return_dataset
from dataloaders.trn.dataset import TSNDataSet


if __name__ == '__main__':
    batch_size = 1
    input_normalise = False 
    input_bgr = True 
#    input_mean = [104, 117, 128]
    input_mean = [117]
    input_std = [1]
    input_frame_length=8

    csv_path = "data/something-something-v1/splits_frames_sorted/val.csv"
    dataset_1 = FramesSparsesampleDataset(csv_path, "val",
            input_frame_length, frame_start_idx=1, test_num_spatial_crops=5,
            mean = input_mean, std = input_std,
            normalise = input_normalise, bgr=input_bgr)

    scale_size = 256
    crop_size = 224
    categories, train_list, val_list, root_path, prefix = return_dataset("something", "RGB")
    dataset_2 = TSNDataSet(root_path, val_list, num_segments=input_frame_length,
                   new_length=1,
                   modality="RGB",
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       GroupNormalize(input_mean, input_std)
                   ]))

    dataloader_1 = iter(torch.utils.data.DataLoader(dataset_1, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0, pin_memory=True, drop_last=False))
    dataloader_2 = iter(torch.utils.data.DataLoader(dataset_2, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0, pin_memory=True, drop_last=False))

    if True:
        inputs, uids, labels, _, _, _, _ = next(dataloader_1)
        # N, C, F, H, W -> N, FC, H, W
        inputs = inputs.permute((0,2,1,3,4)).reshape((batch_size, input_frame_length * 3, crop_size, crop_size))
        inputs2, labels2 = next(dataloader_2)

        print(torch.round(inputs - inputs2))
        print(torch.max(torch.round(inputs - inputs2)))
        print(labels - labels2)
        print(inputs.shape)
        print(inputs2.shape)

        inputs *= torch.tensor(input_std)
        inputs += torch.tensor(input_mean)

        for batch_idx, video in enumerate(inputs):
            for frame_idx, frame in enumerate(video):
                im = Image.fromarray(frame.numpy().astype(np.uint8))
                im.save('1/batch{}_{}.jpg'.format(batch_idx, frame_idx))

        inputs2 *= torch.tensor(input_std)
        inputs2 += torch.tensor(input_mean)
        for batch_idx, video in enumerate(inputs2):
            for frame_idx, frame in enumerate(video):
                im = Image.fromarray(frame.numpy().astype(np.uint8))
                im.save('2/batch{}_{}.jpg'.format(batch_idx, frame_idx))

    if False:
        inputs, uids, labels, _, _, _, _ = next(dataloader_1)

        # N, C, F, H, W -> N, F, H, W, C
        inputs = inputs.permute((0,2,3,4,1))
        inputs *= torch.tensor(input_std)
        inputs += torch.tensor(input_mean)

        for batch_idx, video in enumerate(inputs):
            for frame_idx, frame in enumerate(video):
                im = Image.fromarray(frame.numpy().astype(np.uint8))
                im.save('1/batch{}_{}.jpg'.format(batch_idx, frame_idx))
        print(uids, labels)

        inputs, labels = next(dataloader_2)
        # N, FC, H, W -> N, F, C, H, W
        inputs = inputs.view((batch_size, input_frame_length, 3, crop_size, crop_size))
        # N, F, C, H, W -> N, F, H, W, C
        inputs = inputs.permute((0,1,3,4,2))
        inputs *= torch.tensor(input_std)
        inputs += torch.tensor(input_mean)
        for batch_idx, video in enumerate(inputs):
            for frame_idx, frame in enumerate(video):
                im = Image.fromarray(frame.numpy().astype(np.uint8))
                im.save('2/batch{}_{}.jpg'.format(batch_idx, frame_idx))
        print(labels)

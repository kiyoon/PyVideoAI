import torch
import numpy as np
from PIL import Image

from dataloaders.frames_sparsesample_detection_dataset import FramesSparsesampleDetectionDataset


if __name__ == '__main__':
    batch_size = 1
    input_normalise = False 
    input_bgr = True 
#    input_mean = [104, 117, 128]
    input_mean = [117]
    input_std = [1]
    input_frame_length=8

    csv_path = "data/something-something-v1/splits_frames/val.csv"
    dataset_1 = FramesSparsesampleDetectionDataset(csv_path, "val",
            input_frame_length, '/storage/detectron2_output/ssv1_100boxes_part', frame_start_idx=1, test_num_spatial_crops=5,
            mean = input_mean, std = input_std,
            normalise = input_normalise, bgr=input_bgr)

    dataloader_1 = iter(torch.utils.data.DataLoader(dataset_1, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0, pin_memory=True, drop_last=False))

    if False:
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

    if True:
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


import torch
from PIL import Image
import numpy as np


#from dataloaders.DALI_video_loader import DALILoader
from dataloaders.video_classification_dataset import VideoClassificationDataset
from dataloaders.video_sparsesample_dataset import VideoSparsesampleDataset
from dataloaders.frames_sparsesample_dataset import FramesSparsesampleDataset


if __name__ == '__main__':
    batch_size = 4
    input_frame_length = 16
    input_frame_stride = 1
    input_target_fps = 15000 / 1001     # 14.985014985014985
    mean = 0.45
    std = 0.225
    # Dataset
#    val_dataset = VideoClassificationDataset("/home/kiyoon/datasets/EPIC_KITCHENS_2018/epic_list/train.csv", "train",
#        input_frame_length, input_frame_stride, target_fps=input_target_fps, enable_multi_thread_decode=False, decoding_backend='pyav')
#    val_dataset = VideoSparsesampleDataset("/home/kiyoon/datasets/EPIC_KITCHENS_2018/epic_list/train.csv", "train",
#        8, target_fps=input_target_fps, enable_multi_thread_decode=False, decoding_backend='pyav')
    val_dataset = FramesSparsesampleDataset("/home/kiyoon/datasets/EPIC_KITCHENS_2018/epic_list_frames/train.csv", "train", 8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=4, pin_memory=False, drop_last=False)

    val_dataloader_it = iter(val_dataloader)
    #data = next(val_dataloader_it)
    #data = next(val_dataloader_it)
    #data = next(val_dataloader_it)
    #data = next(val_dataloader_it)
    data = next(val_dataloader_it)
    inputs, uids, labels, _, frame_indices, _ = data
    inputs = inputs.numpy()
    print(uids)
    print(frame_indices)

    inputs = inputs.transpose((0,2,3,4,1))      # B, F, H, W, C
    batch_array = (inputs * std + mean) * 255
    batch_array = batch_array.astype(np.uint8)
    print(batch_array.shape)

    for frame_num, img_array in enumerate(batch_array[2]):
    #img_array = batch_array[0, 0]
        img = Image.fromarray(img_array)
        img.save('{:02d}.png'.format(frame_num))

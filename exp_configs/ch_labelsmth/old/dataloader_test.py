import dataset_configs, model_configs
dataset_cfg = dataset_configs.load_cfg('cater_task2_cameramotion')
model_cfg = model_configs.load_cfg('tsm_resnet50_flow', 'epic100')
import os
import pickle
import torch
import time

input_frame_length = 8
crop_size = 224
train_jitter_min = 224
train_jitter_max = 336
val_scale = 256
val_num_spatial_crops = 1
test_scale = 256
test_num_spatial_crops = 10 if dataset_cfg.horizontal_flip else 1

sample_index_code = 'pyvideoai'
#clip_grad_max_norm = 5

input_type = 'RGB' # RGB / flowrr / flowrg

train_label_type = 'epic100_original'    # epic100_original, 5neighbours

from pyvideoai.dataloaders import FramesSparsesampleDataset
def _dataloader_shape_to_model_input_shape(inputs):
    return model_cfg.NCTHW_to_model_input_shape(inputs)

def get_input_reshape_func(split):
    '''
    if split == 'train':
        return _dataloader_shape_to_model_input_shape
    elif split == 'val':
        return _dataloader_shape_to_model_input_shape
    elif split == 'multicropval':
        return _dataloader_shape_to_model_input_shape
    else:
        raise ValueError(f'Unknown split: {split}')
    '''
    return _dataloader_shape_to_model_input_shape

def _get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]

    if split.startswith('multicrop'):
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops
    else:
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops

    path_prefix=dataset_cfg.frames_dir
    if input_type == 'RGB':
        _input_frame_length = input_frame_length * 5
        flow = None
        flow_neighbours = 5
        flow_folder_x = 'u'
        flow_folder_y = 'v'
    elif input_type == 'flowrr':
        _input_frame_length = input_frame_length
        flow = 'RR'
        flow_neighbours = 5
        flow_folder_x = 'u'
        flow_folder_y = 'v'
    elif input_type == 'flowrg':
        _input_frame_length = input_frame_length
        flow = 'RG'
        flow_neighbours = 5
        flow_folder_x = 'u'
        flow_folder_y = 'v'
    else:
        raise ValueError(f'Wrong input_type {input_type}')

    return FramesSparsesampleDataset(csv_path, mode,
            _input_frame_length,
            train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
            train_horizontal_flip=dataset_cfg.horizontal_flip,
            test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
            crop_size=crop_size,
            mean = model_cfg.input_mean,
            std = model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
            greyscale=False,
            path_prefix=path_prefix,
            sample_index_code=sample_index_code,
            flow = flow,
            flow_neighbours = flow_neighbours,
            flow_folder_x = flow_folder_x,
            flow_folder_y = flow_folder_y,
            )


def get_torch_dataset(split):
    split_dir = dataset_cfg.frames_split_file_dir
    csv_path = os.path.join(split_dir, dataset_cfg.split_file_basename[split])

    return _get_torch_dataset(csv_path, split)



def main():
    train_dataset = get_torch_dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=0, pin_memory=True)
    start_time = time.time()
    for idx, data in enumerate(train_dataloader):
        print(data[0].shape)
        if idx == 10:
            break
    elapsed_time = time.time() - start_time
    print(elapsed_time)

if __name__ == '__main__':
    main()

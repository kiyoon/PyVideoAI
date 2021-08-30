_exec_relative_('sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py')
from pyvideoai.dataloaders.video_sparsesample_dataset import VideoSparsesampleDataset

def _get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]
    if split == 'val':
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops
    else:
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops
    return VideoSparsesampleDataset(csv_path, mode,
            input_frame_length*3 if sampling_mode == 'GreyST' else input_frame_length, 
            train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
            train_horizontal_flip=dataset_cfg.horizontal_flip,
            test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
            crop_size=crop_size,
            mean = [model_cfg.input_mean[0]] if greyscale else model_cfg.input_mean,
            std = [model_cfg.input_std[0]] if greyscale else model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
            greyscale=greyscale,
            path_prefix=dataset_cfg.video_dir,
            sample_index_code=sample_index_code,
            )

def get_torch_dataset(split):

    mode = dataset_cfg.split2mode[split]
    csv_path = os.path.join(dataset_cfg.video_split_file_dir, dataset_cfg.split_file_basename[split])

    return _get_torch_dataset(csv_path, split)

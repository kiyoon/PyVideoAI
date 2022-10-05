"""
Example execution order:
```
_exec_relative_('../sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py')
_exec_relative_('onlytimecantell_temporal_base.py')
```

The only difference is `path_prefix`

"""

from pyvideoai.config import DATA_DIR

def _get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]
    if split == 'val':
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops
    else:
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops

    if sampling_mode == 'GreyST':
        sample_frame_length = input_frame_length * 3
    elif sampling_mode == 'TCPlus2':
        sample_frame_length = input_frame_length + 2
    elif sampling_mode in ['TC', 'RGB', 'GreyOnly']:
        sample_frame_length = input_frame_length
    else:
        raise ValueError(f'Unknown {sampling_mode = }. Should be RGB, TC, TCPlus2, or GreyST')

    return FramesSparsesampleDataset(csv_path, mode,
            sample_frame_length,
            train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
            train_horizontal_flip=dataset_cfg.horizontal_flip,
            test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
            crop_size=crop_size,
            mean = [model_cfg.input_mean[0]] if greyscale else model_cfg.input_mean,
            std = [model_cfg.input_std[0]] if greyscale else model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
            greyscale=greyscale,
            path_prefix=DATA_DIR,
            sample_index_code=sample_index_code,
            )

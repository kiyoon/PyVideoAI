_exec_relative_('../sparsesample_onehot_RGB_crop224_8frame_largejit_plateau.py')
_exec_relative_('../tsm_resnet50_nopartialbn_base.py')

base_learning_rate = 5e-6


pretrained_path = '/home/kiyoon/storage/experiments_ais2/experiments_labelsmooth/epic100_verb/tsm_resnet50_nopartialbn/onehot/version_008/weights/best.pth'
import torch
def load_pretrained(model: torch.nn.Module):
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    del checkpoint

    # freeze parameters
    for param in model.base_model.parameters():
        param.requires_grad = False

from pyvideoai.dataloaders.video_sparsesample_dataset import VideoSparsesampleDataset
import pickle
def _get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]

    if split == 'val':
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops
    else:
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops

    if split == 'train':
        pickle_path = '/home/kiyoon/storage/5-neighbours-from-features_epoch_0020_traindata_testmode_oneclip.pkl'
        with open(pickle_path, 'rb') as f:
            d = pickle.load(f)
        video_ids, soft_labels = d['query_ids'], d['soft_labels']
        video_id_to_label = {}
        for video_id, soft_label in zip(video_ids, soft_labels):
            video_id_to_label[video_id] = soft_label

    else:
        video_id_to_label = None    # single label

    return VideoSparsesampleDataset(csv_path, mode, 
            input_frame_length, 
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
            video_id_to_label = video_id_to_label,

            )

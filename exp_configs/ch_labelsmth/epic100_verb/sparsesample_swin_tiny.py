import os

import torch
from pyvideoai.dataloaders import GulpSparsesampleDataset
from functools import lru_cache
import logging
logger = logging.getLogger(__name__)

input_frame_length = 8
input_type = 'gulp_rgb'  # gulp_rgb / gulp_flow
flow_neighbours = 5

num_epochs = int(os.getenv('VAI_NUM_EPOCHS', default=1000))

#batch_size = 8  # per process (per GPU)
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    if input_type in ['gulp_rgb', 'gulp_flow']:
        divide_batch_size = 1
    else:
        raise ValueError(f'Wrong input_type {input_type}')

    devices = list(range(torch.cuda.device_count()))
    vram = min([torch.cuda.get_device_properties(device).total_memory for device in devices])

    if vram > 20e+9:
        return input_frame_length * 4 // divide_batch_size
    elif vram > 10e+9:
        return input_frame_length * 2 // divide_batch_size
    return input_frame_length // divide_batch_size

def val_batch_size():
    return batch_size() if callable(batch_size) else batch_size

crop_size = 224
train_jitter_min = 224
train_jitter_max = 336
val_scale = 256
val_num_spatial_crops = 1
test_scale = 256
test_num_spatial_crops = 10 if dataset_cfg.horizontal_flip else 1

early_stop_patience = 20        # or None to turn off

sample_index_code = 'pyvideoai'
#clip_grad_max_norm = 5

pretrained = 'kinetics400'      # kinetics400 / random

base_learning_rate = float(os.getenv('VAI_BASELR', 5e-6))      # when batch_size == 1 and #GPUs == 1

loss_type = 'crossentropy'  # crossentropy


#### OPTIONAL
def get_criterion(split):
    if loss_type == 'crossentropy':
        return torch.nn.CrossEntropyLoss()
    else:
        return ValueError(f'Wrong loss type: {loss_type}')


#def epoch_start_script(epoch, exp, args, rank, world_size, train_kit):


# optional
def get_optim_policies(model):
    """
    You can set different learning rates on different blocks of the network.
    Refer to `get_optim_policies()` in pyvideoai/models/epic/tsn.py
    """
    return model_cfg.get_optim_policies(model)


from pyvideoai.utils.early_stopping import min_value_within_lastN, best_value_within_lastN
# optional
def early_stopping_condition(exp, metric_info):
    if early_stop_patience is None:
        return False

    if not min_value_within_lastN(exp.summary['val_loss'], early_stop_patience):
        best_metric_fieldname = metric_info['best_metric_fieldname']
        best_metric_is_better = metric_info['best_metric_is_better_func']
        if not best_value_within_lastN(exp.summary[best_metric_fieldname], early_stop_patience, best_metric_is_better):
            logger.info(f"Validation loss and {best_metric_fieldname} haven't gotten better for {early_stop_patience} epochs. Stopping training..")
            return True

    return False


from pyvideoai.utils.distributed import get_world_size
def optimiser(params):
    """
    LR should be proportional to the total batch size.
    When distributing, LR should be multiplied by the number of processes (# GPUs)
    Thus, LR = base_LR * batch_size_per_proc * (num_GPUs**2)
    """

    batchsize = batch_size() if callable(batch_size) else batch_size
    world_size = get_world_size()

    max_lr = base_learning_rate * 16 * (4**2)
    learning_rate = min(base_learning_rate * batchsize * (world_size**2), max_lr)

    return torch.optim.AdamW(params, lr = learning_rate, betas = (0.9, 0.999), weight_decay = 0.02)


from pyvideoai.utils.lr_scheduling import GradualWarmupScheduler
def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    if base_learning_rate < 1e-5:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.
    else:
        after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.

        return GradualWarmupScheduler(optimiser, multiplier=1, total_epoch=10, after_scheduler=after_scheduler)


def load_model():
    if input_type == 'gulp_rgb':
        input_channel_num = 3
    elif input_type == 'gulp_flow':
        input_channel_num = 2 * flow_neighbours
    else:
        raise ValueError(f'Wrong {input_type = }')

    return model_cfg.load_model(num_classes = dataset_cfg.num_classes,
            input_frame_length = input_frame_length,
            input_channel_num = input_channel_num,
            )


# If you need to extract features, use this. It can be defined in model_cfg too.
#def feature_extract_model(model):
#    from torch.nn import Module
#    class FeatureExtractModel(Module):
#        def __init__(self, model):
#            super().__init__()
#            self.model = model
#        def forward(self, x):
#            return self.model.features(x)
#
#    return FeatureExtractModel(model)

# optional
def load_pretrained(model):
    if pretrained == 'kinetics400':
        model_cfg.load_pretrained_kinetics400(model)
    elif pretrained == 'random':
        pass
    else:
        raise ValueError(f'Wrong {pretrained = }')


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


def _sparse_unpack_data(data):
    '''
    From dataloader returning values to (inputs, uids, labels, [reserved]) format
    '''
    inputs, uids, labels, spatial_idx, _, frame_indices = data
    return inputs, uids, labels, {'spatial_idx': spatial_idx, 'temporal_idx': -1 * torch.ones_like(labels), 'frame_indices': frame_indices}

def _dense_unpack_data(data):
    '''
    From dataloader returning values to (inputs, uids, labels, [reserved]) format
    '''
    inputs, uids, labels, spatial_idx, temporal_idx, _, frame_indices = data
    return inputs, uids, labels, {'spatial_idx': spatial_idx, 'temporal_idx': temporal_idx, 'frame_indices': frame_indices}

def get_data_unpack_func(split):
    if split in ['multicropval_strided', 'multicroptrain_strided']:
        return _dense_unpack_data
    else:
        return _sparse_unpack_data


def _get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]

    if split.startswith('multicrop'):
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops
    else:
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops

    video_id_to_label = None
    if input_type == 'gulp_rgb':
        gulp_dir_path = os.path.join(dataset_cfg.dataset_root, dataset_cfg.gulp_rgb_dirname[split])

        return GulpSparsesampleDataset(csv_path, mode,
                input_frame_length, gulp_dir_path,
                train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
                train_horizontal_flip=dataset_cfg.horizontal_flip,
                test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
                crop_size=crop_size,
                mean = model_cfg.input_mean,
                std = model_cfg.input_std,
                normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
                greyscale=False,
                sample_index_code=sample_index_code,
                processing_backend = 'pil',
                video_id_to_label = video_id_to_label,
                )
    elif input_type == 'gulp_flow':
        gulp_dir_path = os.path.join(dataset_cfg.dataset_root, dataset_cfg.gulp_flow_dirname[split])

        return GulpSparsesampleDataset(csv_path, mode,
                input_frame_length, gulp_dir_path,
                train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
                train_horizontal_flip=dataset_cfg.horizontal_flip,
                test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
                crop_size=crop_size,
                mean = model_cfg.input_mean,
                std = model_cfg.input_std,
                normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
                greyscale=False,
                sample_index_code=sample_index_code,
                processing_backend = 'pil',
                flow = 'grey',
                flow_neighbours = flow_neighbours,
                video_id_to_label = video_id_to_label,
                )
    else:
        raise ValueError(f'Wrong input_type {input_type}')



@lru_cache
def get_torch_dataset(split):
    if input_type == 'gulp_rgb':
        split_dir = dataset_cfg.gulp_rgb_split_file_dir
    elif input_type == 'gulp_flow':
        split_dir = dataset_cfg.gulp_flow_split_file_dir
    else:
        raise ValueError(f'Wrong input_type {input_type}')
    csv_path = os.path.join(split_dir, dataset_cfg.split_file_basename[split])

    return _get_torch_dataset(csv_path, split)


"""
OPTIONAL: change metrics, plotting figures and reporting text.
when you resume from checkpoint, load optimiser/scheduler state?
Default values are True.
"""
#load_optimiser_state = True
#load_scheduler_state = True

"""
OPTIONAL but important: configure DDP and AMP
Uncommenting following lines can speed up the training but also can introduce exceptions depending on your model.
Recommeded to change this settings in model_configs
"""
#ddp_find_unused_parameters = False
#use_amp = True

"""
## For both train & val
Changing last_activation and leaving metrics/predictions_gatherers commented out will still change the default metrics and predictions_gatherers' activation function
"""
last_activation = 'softmax'   # or, you can pass a callable function like `torch.nn.Softmax(dim=1)`

"""
## For training, (tools/run_train.py)
how to calculate metrics
"""
from pyvideoai.metrics.accuracy import ClipAccuracyMetric, VideoAccuracyMetric
from pyvideoai.metrics.grouped_class_accuracy import ClipGroupedClassAccuracyMetric
from exp_configs.ch_labelsmth.epic100_verb.read_multilabel import read_multiverb, get_val_holdout_set
from video_datasets_api.epic_kitchens_100.read_annotations import get_verb_uid2label_dict
video_id_to_multilabel = read_multiverb(dataset_cfg.narration_id_to_video_id)
_, epic_video_id_to_label = get_verb_uid2label_dict(dataset_cfg.annotations_root)
holdout_video_id_to_label = get_val_holdout_set(epic_video_id_to_label, video_id_to_multilabel)

best_metric = ClipAccuracyMetric(video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval')
metrics = {'train': [ClipAccuracyMetric(),
            ClipGroupedClassAccuracyMetric([dataset_cfg.head_classes, dataset_cfg.tail_classes], ['head', 'tail'])],
        'val': [best_metric,
            ClipAccuracyMetric(topk=(5,), video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval'),
            ClipGroupedClassAccuracyMetric([dataset_cfg.head_classes, dataset_cfg.tail_classes], ['head', 'tail'], video_id_to_label = holdout_video_id_to_label, video_id_to_label_missing_action = 'skip', split='holdoutval'),
            ],
        'traindata_testmode': [ClipAccuracyMetric()],
        'trainpartialdata_testmode': [ClipAccuracyMetric()],
        'multicropval': [ClipAccuracyMetric(), VideoAccuracyMetric(topk=(1,5), activation=last_activation)],
        }

"""
## For validation, (tools/run_val.py)
how to gather predictions when --save_predictions is set
"""
from pyvideoai.metrics.metric import ClipPredictionsGatherer, VideoPredictionsGatherer
predictions_gatherers = {'val': ClipPredictionsGatherer(),
        'traindata_testmode': ClipPredictionsGatherer(),
        'trainpartialdata_testmode': ClipPredictionsGatherer(),
        'multicropval': VideoPredictionsGatherer(),
        }

"""How will you plot"""
#from pyvideoai.visualisations.metric_plotter import DefaultMetricPlotter
#metric_plotter = DefaultMetricPlotter()
#from pyvideoai.visualisations.telegram_reporter import DefaultTelegramReporter
#telegram_reporter = DefaultTelegramReporter()

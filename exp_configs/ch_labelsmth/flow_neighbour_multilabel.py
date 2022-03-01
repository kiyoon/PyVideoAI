import os
import pickle

from pyvideoai.dataloaders import FramesSparsesampleDataset
from pyvideoai.utils.losses.proselflc import ProSelfLC, InstableCrossEntropy
from pyvideoai.utils.losses.loss import LabelSmoothCrossEntropyLoss
from pyvideoai.utils.losses.softlabel import SoftlabelRegressionLoss
from pyvideoai.utils import loader

import torch

#batch_size = 8  # per process (per GPU)
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    devices=list(range(torch.cuda.device_count()))
    vram = min([torch.cuda.get_device_properties(device).total_memory for device in devices])
    if vram > 20e+9:
        return 32
    elif vram > 10e+9:
        return 16
    return 8

def val_batch_size():
    return batch_size() if callable(batch_size) else batch_size

input_frame_length = 8
crop_size = 224
train_jitter_min = 224
train_jitter_max = 336
val_scale = 256
val_num_spatial_crops = 1
test_scale = 256
test_num_spatial_crops = 10 if dataset_cfg.horizontal_flip else 1

greyscale=False
sample_index_code = 'pyvideoai'
#clip_grad_max_norm = 5

base_learning_rate = 5e-6      # when batch_size == 1 and #GPUs == 1

train_label_type = 'epic100_original'    # epic100_original, 5neighbours
loss_type = 'crossentropy'   # soft_regression, crossentropy, labelsmooth, proselflc

labelsmooth_factor = 0.1
#proselflc_total_time = 2639 * 60 # 60 epochs
proselflc_total_time = 263 * 40 # 60 epochs
proselflc_exp_base = 1.


#### OPTIONAL
def get_criterion(split):
    if loss_type == 'labelsmooth':
        return LabelSmoothCrossEntropyLoss(smoothing=labelsmooth_factor)
    elif loss_type == 'proselflc':
        if split == 'train':
            return ProSelfLC(proselflc_total_time, proselflc_exp_base)
        else:
            return torch.nn.CrossEntropyLoss()
    elif loss_type == 'soft_regression':
        return SoftlabelRegressionLoss()
    else:
        return torch.nn.CrossEntropyLoss()

from torch.utils.data.distributed import DistributedSampler
import pyvideoai.utils.distributed as du
train_testmode_dataloader = None
def epoch_start_script(epoch, exp, args, rank, world_size, train_kit):
    feature_extract_split = 'trainpartialdata_testmode'

    if train_testmode_dataloader is None:
        train_testmode_dataset = get_torch_dataset(feature_extract_split)
        sampler = DistributedSampler(train_testmode_dataset, shuffle=False) if world_size > 1 else None
        train_testmode_dataloader = torch.utils.data.DataLoader(train_testmode_dataset, batch_size=val_batch_size(), shuffle=False, sampler=sampler, num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=False, worker_init_fn = du.seed_worker)

    data_unpack_func = get_data_unpack_func(feature_extract_split)
    input_reshape_func = get_input_reshape_func(feature_extract_split)
    feature_data, _, _, _, eval_log_str = extract_features(train_kit["model"], train_testmode_dataloader, data_unpack_func, dataset_cfg.num_classes, feature_extract_split, rank, world_size, input_reshape_func=input_reshape_func, refresh_period=args.refresh_period)
    return None

# optional
def get_optim_policies(model):
    """
    You can set different learning rates on different blocks of the network.
    Refer to `get_optim_policies()` in pyvideoai/models/epic/tsn.py
    """
    return model_cfg.get_optim_policies(model)

import logging
logger = logging.getLogger(__name__)
from pyvideoai.utils.early_stopping import min_value_within_lastN, best_value_within_lastN
# optional
def early_stopping_condition(exp, metric_info):
    patience=20
    if not min_value_within_lastN(exp.summary['val_loss'], patience):
        best_metric_fieldname = metric_info['best_metric_fieldname']
        best_metric_is_better = metric_info['best_metric_is_better_func']
        if not best_value_within_lastN(exp.summary[best_metric_fieldname], patience, best_metric_is_better):
            logger.info(f"Validation loss and {best_metric_fieldname} haven't gotten better for {patience} epochs. Stopping training..")
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

    return torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)

from pyvideoai.utils.lr_scheduling import ReduceLROnPlateauMultiple, GradualWarmupScheduler
def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.

    #return GradualWarmupScheduler(optimiser, multiplier=1, total_epoch=10, after_scheduler=after_scheduler) 
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length)

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
#def load_pretrained(model):
#    loader.model_load_weights_GPU(model, pretrained_path)

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


def _unpack_data(data):
    '''
    From dataloader returning values to (inputs, uids, labels, [reserved]) format
    '''
    inputs, uids, labels, spatial_idx, _, _ = data
    return inputs, uids, labels, {'spatial_idx': spatial_idx, 'temporal_idx': -1 *torch.ones_like(labels)}


def get_data_unpack_func(split):
    '''
    if split == 'train':
        return _unpack_data
    elif split == 'val':
        return _unpack_data
    elif split == 'multicropval':
        return _unpack_data
    else:
        assert False, 'unknown split'
    '''
    return _unpack_data


def _get_torch_dataset(csv_path, split):
    mode = dataset_cfg.split2mode[split]

    if split.startswith('multicrop'):
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops
    else:
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops

    video_id_to_label = None
    if split == 'train':
        if train_label_type == '5neighbours':
            video_id_to_label = {}
            softlabel_pickle_path = '/home/kiyoon/storage/tsm_flow_neigh/5-neighbours-from-features_epoch_0009_traindata_testmode_oneclip.pkl'
            with open(softlabel_pickle_path, 'rb') as f:
                d = pickle.load(f)
            video_ids, soft_labels = d['query_ids'], d['soft_labels']

            for video_id, soft_label in zip(video_ids, soft_labels):
                video_id_to_label[video_id] = soft_label


    return FramesSparsesampleDataset(csv_path, mode, 
            input_frame_length, 
            train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
            train_horizontal_flip=dataset_cfg.horizontal_flip,
            test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
            crop_size=crop_size,
            mean = [model_cfg.input_mean[0]] if greyscale else model_cfg.input_mean,
            std = [model_cfg.input_std[0]] if greyscale else model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
            greyscale=greyscale,
            path_prefix=dataset_cfg.flowframes_dir,
            sample_index_code=sample_index_code,
            flow = 'RR',
            flow_neighbours = 5,
            flow_folder_x = 'u',
            flow_folder_y = 'v',
            video_id_to_label = video_id_to_label,
            )

def get_torch_dataset(split):
    csv_path = os.path.join(dataset_cfg.flowframes_split_file_dir, dataset_cfg.split_file_basename[split])

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
from pyvideoai.metrics.mean_perclass_accuracy import ClipMeanPerclassAccuracyMetric
from pyvideoai.metrics.grouped_class_accuracy import ClipGroupedClassAccuracyMetric
best_metric = ClipAccuracyMetric()
metrics = {'train': [ClipAccuracyMetric(), ClipMeanPerclassAccuracyMetric(), ClipGroupedClassAccuracyMetric([dataset_cfg.head_classes, dataset_cfg.tail_classes], ['head', 'tail'])],
        'val': [best_metric, ClipMeanPerclassAccuracyMetric(), ClipGroupedClassAccuracyMetric([dataset_cfg.head_classes, dataset_cfg.tail_classes], ['head', 'tail']),
            ClipMultilabelAccuracyMetric(video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip'),
            ClipTop1MultilabelAccuracyMetric(video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip'),
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
predictions_gatherers = {'val': ClipPredictionsGatherer(last_activation),
        'traindata_testmode': ClipPredictionsGatherer(last_activation),
        'trainpartialdata_testmode': ClipPredictionsGatherer(last_activation),
        'multicropval': VideoPredictionsGatherer(last_activation),
        }

"""How will you plot"""
#from pyvideoai.visualisations.metric_plotter import DefaultMetricPlotter
#metric_plotter = DefaultMetricPlotter()
#from pyvideoai.visualisations.telegram_reporter import DefaultTelegramReporter
#telegram_reporter = DefaultTelegramReporter()

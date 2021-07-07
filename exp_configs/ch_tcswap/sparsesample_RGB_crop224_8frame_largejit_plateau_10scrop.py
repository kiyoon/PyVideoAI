import os

from pyvideoai.dataloaders.frames_sparsesample_dataset import FramesSparsesampleDataset

import torch

#batch_size = 8  # per process (per GPU)
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    vram = torch.cuda.get_device_properties(0).total_memory
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
test_num_spatial_crops = 10

#### OPTIONAL
#def criterion():
#    return torch.nn.CrossEntropyLoss()
#
#def epoch_start_script(epoch, exp, args, rank, world_size, train_kit):
#    return None

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
    base_learning_rate = 1e-5      # when batch_size == 1 and #GPUs == 1

    batchsize = batch_size() if callable(batch_size) else batch_size
    world_size = get_world_size()
    learning_rate = base_learning_rate * batchsize * (world_size**2)

    return torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)

def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    #return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0 = 100 * iters_per_epoch, T_mult = 1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    #return torch.optim.lr_scheduler.StepLR(optimiser, step_size = 50 * iters_per_epoch, gamma = 0.1, last_epoch=last_epoch)     # Here, last_epoch means last iteration.
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.
    #return None

def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length)

# optional
#def load_pretrained(model):
#    return

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
    return inputs, uids, labels, spatial_idx


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
    if split == 'val':
        _test_scale = val_scale
        _test_num_spatial_crops = val_num_spatial_crops
    else:
        _test_scale = test_scale
        _test_num_spatial_crops = test_num_spatial_crops
    return FramesSparsesampleDataset(csv_path, mode,
            input_frame_length, 
            train_jitter_min = train_jitter_min, train_jitter_max=train_jitter_max,
            test_scale = _test_scale, test_num_spatial_crops=_test_num_spatial_crops,
            crop_size=crop_size,
            mean = model_cfg.input_mean, std = model_cfg.input_std,
            normalise = model_cfg.input_normalise, bgr=model_cfg.input_bgr,
            path_prefix=dataset_cfg.frames_dir)

def get_torch_dataset(split):

    mode = dataset_cfg.split2mode[split]
    csv_path = os.path.join(dataset_cfg.frames_split_file_dir, dataset_cfg.split_file_basename[split])

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
#last_activation = 'softmax'   # or, you can pass a callable function like `torch.nn.Softmax(dim=1)`

"""
## For training, (tools/run_train.py)
how to calculate metrics
"""
#from pyvideoai.metrics.accuracy import ClipAccuracyMetric, VideoAccuracyMetric
#best_metric = ClipAccuracyMetric()
#metrics = {'train': [ClipAccuracyMetric()],
#        'val': [best_metric],
#        'multicropval': [ClipAccuracyMetric(), VideoAccuracyMetric(topk=(1,5), activation=last_activation)],
#        }
#
"""
## For validation, (tools/run_val.py)
how to gather predictions when --save_predictions is set
"""
#from pyvideoai.metrics.metric import ClipPredictionsGatherer, VideoPredictionsGatherer
#predictions_gatherers = {'val': ClipPredictionsGatherer(last_activation),
#        'multicropval': VideoPredictionsGatherer(last_activation),
#        }
#
"""How will you plot"""
#from pyvideoai.visualisations.metric_plotter import DefaultMetricPlotter
#metric_plotter = DefaultMetricPlotter()
#from pyvideoai.visualisations.telegram_reporter import DefaultTelegramReporter
#telegram_reporter = DefaultTelegramReporter()

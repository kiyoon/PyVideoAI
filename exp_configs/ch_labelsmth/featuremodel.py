from pyvideoai.dataloaders.feature_dataset import FeatureDataset
import pickle
import numpy as np
from pyvideoai.utils.losses.softlabel import SoftlabelRegressionLoss
import torch
import os

#batch_size = 8  # per process (per GPU)
def batch_size():
    '''batch_size can be either integer or function returning integer.
    '''
    def default():
        devices=list(range(torch.cuda.device_count()))
        vram = min([torch.cuda.get_device_properties(device).total_memory for device in devices])
        if vram > 20e+9:
            return 2048
        elif vram > 10e+9:
            return 1024
        return 64
    return int(os.getenv('PYVIDEOAI_BATCHSIZE', default()))

def val_batch_size():
    return batch_size() if callable(batch_size) else batch_size

train_label_type = '5neighbours'    # epic100_original, 5neighbours
loss_type = 'soft_regression'   # soft_regression, crossentropy

#clip_grad_max_norm = 20
learning_rate = float(os.getenv('PYVIDEOAI_LR', 5e-6))      # when batch_size == 1 and #GPUs == 1

#### OPTIONAL
def get_criterion(split):
    if loss_type == 'soft_regression':
        return SoftlabelRegressionLoss()
    elif loss_type == 'crossentropy':
        return torch.nn.CrossEntropyLoss()
#
#def epoch_start_script(epoch, exp, args, rank, world_size, train_kit):
#    return None

# optional
#def get_optim_policies(model):
#    """
#    You can set different learning rates on different blocks of the network.
#    Refer to `get_optim_policies()` in pyvideoai/models/epic/tsn.py
#    """
#    return model_cfg.get_optim_policies(model)

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

    #return torch.optim.SGD(params, lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)
    return torch.optim.Adam(params, lr = learning_rate)

#from pyvideoai.utils.lr_scheduling import GradualWarmupScheduler
def scheduler(optimiser, iters_per_epoch, last_epoch=-1):
    return None
#    after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10, verbose=True)     # NOTE: This special scheduler will ignore iters_per_epoch and last_epoch.

#    return GradualWarmupScheduler(optimiser, multiplier=1, total_epoch=10, after_scheduler=after_scheduler) 

def load_model():
    num_layers = int(os.getenv('PYVIDEOAI_NUMLAYERS', 2))
    num_units = int(os.getenv('PYVIDEOAI_NUMUNITS', 1024))
    return model_cfg.load_model(dataset_cfg.num_classes, 2048, num_layers=num_layers, num_units=num_units)

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
#    return

def _dataloader_shape_to_model_input_shape(inputs):
    return inputs

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
    inputs, uids, labels, _ = data
    return inputs, uids, labels, {'spatial_idx': 0, 'temporal_idx': -1 *torch.ones_like(labels)}


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


def get_torch_dataset(split):

    if split == 'train':

        if train_label_type == '5neighbours':
            softlabel_pickle_path = '/home/kiyoon/storage/tsm_flow_neigh/5-neighbours-from-features_epoch_0009_traindata_testmode_oneclip.pkl'
            with open(softlabel_pickle_path, 'rb') as f:
                d = pickle.load(f)
            video_ids_soft, soft_labels = d['query_ids'], d['soft_labels']
            sort_idx = np.argsort(video_ids_soft)

            video_ids_soft = video_ids_soft[sort_idx]
            labels = soft_labels[sort_idx]

            feature_pickle_path = '/home/kiyoon/storage/experiments_ais2/experiments_labelsmooth/epic100_verb/ch_epic100.tsm_resnet50_flow/onehot/version_002/predictions/features_epoch_0009_traindata_testmode_oneclip.pkl'
            with open(feature_pickle_path, 'rb') as f:
                d = pickle.load(f)

            video_ids, features = d['video_ids'], d['clip_features']
            sort_idx = np.argsort(video_ids)
            video_ids = video_ids[sort_idx]
            features = features[sort_idx]

            assert np.all(video_ids == video_ids_soft)
        elif train_label_type == 'epic100_original':
            feature_pickle_path = '/home/kiyoon/storage/experiments_ais2/experiments_labelsmooth/epic100_verb/ch_epic100.tsm_resnet50_flow/onehot/version_002/predictions/features_epoch_0009_traindata_testmode_oneclip.pkl'
            with open(feature_pickle_path, 'rb') as f:
                d = pickle.load(f)

            video_ids, labels, features = d['video_ids'], d['labels'], d['clip_features']

    elif split == 'val':
        feature_pickle_path = '/home/kiyoon/storage/experiments_ais2/experiments_labelsmooth/epic100_verb/ch_epic100.tsm_resnet50_flow/onehot/version_002/predictions/features_epoch_0009_val_oneclip.pkl'
        with open(feature_pickle_path, 'rb') as f:
            d = pickle.load(f)

        video_ids, labels, features = d['video_ids'], d['labels'], d['clip_features']

    else:
        raise ValueError(f'Only train and val splits are supported. Got {split}')

    return FeatureDataset(video_ids, labels, features)


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
from pyvideoai.metrics.multilabel_accuracy import ClipMultilabelAccuracyMetric
from pyvideoai.metrics.top1_multilabel_accuracy import ClipTop1MultilabelAccuracyMetric
from exp_configs.ch_labelsmth.epic100_verb.read_multilabel import read_multilabel
video_id_to_multilabel = read_multilabel()
best_metric = ClipAccuracyMetric(topk=(1,5))
metrics = {'train': [ClipAccuracyMetric(), ClipMeanPerclassAccuracyMetric(), ClipGroupedClassAccuracyMetric([dataset_cfg.head_classes, dataset_cfg.tail_classes], ['head', 'tail'])],
        'val': [best_metric, ClipMeanPerclassAccuracyMetric(), ClipGroupedClassAccuracyMetric([dataset_cfg.head_classes, dataset_cfg.tail_classes], ['head', 'tail']),
            ClipMultilabelAccuracyMetric(video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip'),
            ClipTop1MultilabelAccuracyMetric(video_id_to_label = video_id_to_multilabel, video_id_to_label_missing_action = 'skip'),
            ],
        'multicropval': [ClipAccuracyMetric(), VideoAccuracyMetric(topk=(1,5), activation=last_activation)],
        }

"""
## For validation, (tools/run_val.py)
how to gather predictions when --save_predictions is set
"""
from pyvideoai.metrics.metric import ClipPredictionsGatherer, VideoPredictionsGatherer
predictions_gatherers = {'val': ClipPredictionsGatherer(last_activation),
        'multicropval': VideoPredictionsGatherer(last_activation),
        }

"""How will you plot"""
#from pyvideoai.visualisations.metric_plotter import DefaultMetricPlotter
#metric_plotter = DefaultMetricPlotter()
from pyvideoai.visualisations.telegram_reporter import DefaultTelegramReporter
telegram_reporter = DefaultTelegramReporter(include_exp_rootdir = True, ignore_figures = True)
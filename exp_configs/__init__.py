import os, sys
import importlib
import glob

import dataset_configs
import model_configs

_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))


def _get_module_name(dataset_name, model_name, experiment_name, channel = ''):
    if channel == '' or channel is None:
        return f'.{dataset_name}.{model_name}-{experiment_name}'
    else:
        return f'.ch_{channel}.{dataset_name}.{model_name}-{experiment_name}'


def _get_file_name(dataset_name, model_name, experiment_name, channel=''):
    """
    Not a full path
    """
    if channel == '' or channel is None:
        return os.path.join(dataset_name, f'{model_name}-{experiment_name}.py')
    else:
        return os.path.join(f'ch_{channel}', dataset_name, f'{model_name}-{experiment_name}.py')

def load_cfg(dataset_name, model_name, experiment_name, dataset_channel='', model_channel='', exp_channel=''):
    cfg = importlib.import_module(_get_module_name(dataset_name, model_name, experiment_name, exp_channel), __name__)
    cfg.dataset_cfg = dataset_configs.load_cfg(dataset_name, dataset_channel)
    cfg.model_cfg = model_configs.load_cfg(model_name, model_channel)
    return cfg

def config_path(dataset_name, model_name, experiment_name, channel=''):
    return os.path.join(_SCRIPT_DIR, _get_file_name(dataset_name, model_name, experiment_name, channel))

#available_configs = [os.path.splitext(os.path.basename(fn))[0] for fn in glob.glob(os.path.join(_SCRIPT_DIR, '*.py'))
#                 if os.path.basename(fn) not in ['__init__.py']]

def list_only_dir(path):
    return [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

available_channels = [os.path.basename(name).replace('ch_', '', 1) for name in list_only_dir(_SCRIPT_DIR)
                 if os.path.basename(name).startswith('ch_')]

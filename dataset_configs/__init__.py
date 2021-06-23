import os, sys
import importlib
import glob

_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))


def load_cfg(dataset_name, channel=''):
    """
    params:
        dataset_name (str): name of the dataset_config file without extension(.py)
        channel (str): which folder to import from. Empty string for default channel (no subfolder). Channel folder should be named in ch_{channel} format.

    return:
        dataset_config module
    """
    if channel == '' or channel is None:
        return importlib.import_module('.' + dataset_name, __name__)
    else:
        return importlib.import_module('.' + dataset_name, f'{__name__}.ch_{channel}')


def config_path(dataset_name, channel=''):
    if channel == '' or channel is None:
        return os.path.join(_SCRIPT_DIR, dataset_name + '.py')
    else:
        return os.path.join(_SCRIPT_DIR, f'ch_{channel}', dataset_name + '.py')

# only default channel
available_datasets = [os.path.splitext(os.path.basename(fn))[0] for fn in glob.glob(os.path.join(_SCRIPT_DIR, '*.py'))
                 if os.path.basename(fn) not in ['__init__.py']]

def list_only_dir(path):
    return [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

available_channels = [os.path.basename(name).replace('ch_', '', 1) for name in list_only_dir(_SCRIPT_DIR)
                 if os.path.basename(name).startswith('ch_')]

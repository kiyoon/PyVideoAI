import os, sys
import importlib
import importlib.util
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

# Legacy
#def load_cfg(dataset_name, model_name, experiment_name, dataset_channel='', model_channel='', exp_channel=''):
#    cfg = importlib.import_module(_get_module_name(dataset_name, model_name, experiment_name, exp_channel), __name__)
#    cfg.dataset_cfg = dataset_configs.load_cfg(dataset_name, dataset_channel)
#    cfg.model_cfg = model_configs.load_cfg(model_name, model_channel)
#    return cfg

exec_relative_code = '''
import os
def _exec_relative_(python_path):    
    """
    Dynamically import (exec) base config code.
    Only relative path should be used, and it will find correct config base recursively.
    """
    with open(os.path.join(*_current_dir_stack_, python_path), 'r') as f:
        config_code = f.read()
        
    _current_dir_stack_.append(os.path.dirname(python_path))
    exec(config_code, globals())    # Adding globals() will make the new variables accessible after the call.
    _current_dir_stack_.pop()
'''

def load_cfg(dataset_name, model_name, experiment_name, dataset_channel='', model_channel='', exp_channel=''):
    """
    Reference: https://stackoverflow.com/questions/55905240/python-dynamically-import-modules-code-from-string-with-importlib

    1. Create an empty module (importlib.util.module_from_spec)
    2. Add dataset_cfg, model_cfg in the module
    3. Add `_exec_relative_()` in the module.
    4. Exec module (exp_config)
        a. Note that `_exec_relative_` can be called. It'll recursively find correct config and exec' them.
            
    Therefore, in the cfg object, you can access `dataset_cfg`, `model_cfg`, `_exec_relative(python_path)`.
    """
    # Create an empty module
    spec = importlib.util.find_spec(_get_module_name(dataset_name, model_name, experiment_name, exp_channel), package=__name__)
    cfg = importlib.util.module_from_spec(spec)

    # Add dataset_cfg, model_cfg to the module
    cfg.dataset_cfg = dataset_configs.load_cfg(dataset_name, dataset_channel)
    cfg.model_cfg = model_configs.load_cfg(model_name, model_channel)

    # Add _exec_relative_() to the module
    # for dynamically adding base config files.
    cfg._current_dir_stack_ = [os.path.dirname(os.path.realpath(cfg.__file__))]
    exec(exec_relative_code, cfg.__dict__)

    # Finally, exec the config module
    spec.loader.exec_module(cfg)


    return cfg

def config_path(dataset_name, model_name, experiment_name, channel=''):
    return os.path.join(_SCRIPT_DIR, _get_file_name(dataset_name, model_name, experiment_name, channel))

#available_configs = [os.path.splitext(os.path.basename(fn))[0] for fn in glob.glob(os.path.join(_SCRIPT_DIR, '*.py'))
#                 if os.path.basename(fn) not in ['__init__.py']]

def list_only_dir(path):
    return [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

available_channels = [os.path.basename(name).replace('ch_', '', 1) for name in list_only_dir(_SCRIPT_DIR)
                 if os.path.basename(name).startswith('ch_')]

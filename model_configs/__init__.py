import os, sys
import importlib
import importlib.util
import glob

from shutil import copy2

_SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))

exec_relative_code = '''
import os
def _exec_relative_(python_path):    
    """
    Dynamically import (exec) base config code.
    Only relative path should be used, and it will find correct config base recursively.
    """
    base_config_path = os.path.realpath(os.path.join(*_current_dir_stack_, python_path))
    _exec_paths_.append(base_config_path)
    with open(base_config_path, 'r') as f:
        config_code = f.read()
        
    _current_dir_stack_.append(os.path.dirname(python_path))
    exec(config_code, globals())    # Adding globals() will make the new variables accessible after the call.
    _current_dir_stack_.pop()
'''
def load_cfg(model_name, channel=''):
    """
    params:
        model_name (str): name of the model_config file without extension(.py)
        channel (str): which folder to import from. Empty string for default channel (no subfolder). Channel folder should be named in ch_{channel} format.

    return:
        model_config module
    """
    # Create an empty module
    if channel == '' or channel is None:
        package = __name__
    else:
        package = f'{__name__}.ch_{channel}'
    spec = importlib.util.find_spec('.' + model_name, package=package)

    if spec is None:
        raise FileNotFoundError((f'Cannot load the model config module: {package}.{model_name}\n'
                f'Maybe the file is missing: {config_path(model_name, channel)}'))

    cfg = importlib.util.module_from_spec(spec)

    # Add _exec_relative_() to the module
    # for dynamically adding base config files.
    cfg._current_dir_stack_ = [os.path.dirname(os.path.realpath(cfg.__file__))]
    cfg._exec_paths_ = []
    exec(exec_relative_code, cfg.__dict__)

    # Finally, exec the config module
    spec.loader.exec_module(cfg)

    return cfg


def copy_cfg_files(cfg, dest_dir):
    """
    Backup the config file and its base config files exec'ed using _exec_relative_().
    """
    src_dir = _SCRIPT_DIR

    for cfg_path in [os.path.realpath(cfg.__file__)] + cfg._exec_paths_:
        rel_path = os.path.relpath(cfg_path, src_dir)
        dest_path = os.path.join(dest_dir, rel_path)

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        copy2(cfg_path, dest_path, follow_symlinks=True)


def wandb_log_cfg_files(cfg):
    """
    Using Weights & Biases (wandb.ai), backup the config file and its base config files exec'ed using _exec_relative_().
    We don't use `log_code()` because it doesn't support symlinks.
    """
    import wandb
    code_artifact = wandb.Artifact(__name__, type='configs')
    src_dir = _SCRIPT_DIR

    for cfg_path in [os.path.realpath(cfg.__file__)] + cfg._exec_paths_:
        rel_path = os.path.relpath(cfg_path, src_dir)
        code_artifact.add_file(cfg_path, name=rel_path)
    wandb.log_artifact(code_artifact)


def config_path(model_name, channel=''):
    if channel == '' or channel is None:
        return os.path.join(_SCRIPT_DIR, model_name + '.py')
    else:
        return os.path.join(_SCRIPT_DIR, f'ch_{channel}', model_name + '.py')


# only default channel
available_models = [os.path.splitext(os.path.basename(fn))[0] for fn in glob.glob(os.path.join(_SCRIPT_DIR, '*.py'))
                 if os.path.basename(fn) not in ['__init__.py']]

def list_only_dir(path):
    return [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]

available_channels = [os.path.basename(name).replace('ch_', '', 1) for name in list_only_dir(_SCRIPT_DIR)
                 if os.path.basename(name).startswith('ch_')]

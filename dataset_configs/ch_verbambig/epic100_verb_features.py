"""
Use Resnet50 features instead of videos.
"""

_exec_relative_('../epic100_verb.py')

RGB_features_pickle_path = {'train': os.path.join(dataset_root, 'features', 'RGB', 'features_epoch_0020_traindata_testmode_oneclip.pkl'),
                            'val': os.path.join(dataset_root, 'features', 'RGB', 'features_epoch_0020_val_oneclip.pkl'),
        }

flow_features_pickle_path = {'train': os.path.join(dataset_root, 'features', 'flow', 'features_epoch_0009_traindata_testmode_oneclip.pkl'),
                            'val': os.path.join(dataset_root, 'features', 'flow', 'features_epoch_0009_val_oneclip.pkl'),
        }

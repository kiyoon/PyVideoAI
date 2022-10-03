"""
Use Resnet50 features instead of videos.
"""

_exec_relative_('confusing_hmdb_102.py')

RGB_features_pickle_path = {}
RGB_features_pickle_path[1] = {'train': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'tsm_resnet50_nopartialbn', 'split1', 'features_epoch_0099_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'tsm_resnet50_nopartialbn', 'split1', 'features_epoch_0099_val_oneclip.pkl'),
                                  }
RGB_features_pickle_path[2] = {'train': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'tsm_resnet50_nopartialbn', 'split2', 'features_epoch_0114_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'tsm_resnet50_nopartialbn', 'split2', 'features_epoch_0114_val_oneclip.pkl'),
                                  }
RGB_features_pickle_path[3] = {'train': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'tsm_resnet50_nopartialbn', 'split3', 'features_epoch_0107_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'tsm_resnet50_nopartialbn', 'split3', 'features_epoch_0107_val_oneclip.pkl'),
                                  }

flow_features_pickle_path = {}
flow_features_pickle_path[1] = {'train': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'ch_epic.tsm_resnet50_flow', 'split1', 'features_epoch_0070_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'ch_epic.tsm_resnet50_flow', 'split1', 'features_epoch_0070_val_oneclip.pkl'),
                                  }
flow_features_pickle_path[2] = {'train': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'ch_epic.tsm_resnet50_flow', 'split2', 'features_epoch_0064_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'ch_epic.tsm_resnet50_flow', 'split2', 'features_epoch_0064_val_oneclip.pkl'),
                                  }
flow_features_pickle_path[3] = {'train': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'ch_epic.tsm_resnet50_flow', 'split3', 'features_epoch_0080_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'confusing_hmdb_102', 'ch_epic.tsm_resnet50_flow', 'split3', 'features_epoch_0080_val_oneclip.pkl'),
                                  }

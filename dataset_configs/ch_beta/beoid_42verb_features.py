"""
Use Resnet50 features instead of videos.
"""

_exec_relative_('beoid_42verb.py')

RGB_features_pickle_path = {}
RGB_features_pickle_path[0] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split0', 'features_epoch_0021_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split0', 'features_epoch_0021_val_oneclip.pkl'),
                                  }
RGB_features_pickle_path[1] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split1', 'features_epoch_0011_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split1', 'features_epoch_0011_val_oneclip.pkl'),
                                  }
RGB_features_pickle_path[2] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split2', 'features_epoch_0024_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split2', 'features_epoch_0024_val_oneclip.pkl'),
                                  }
RGB_features_pickle_path[3] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split3', 'features_epoch_0048_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split3', 'features_epoch_0048_val_oneclip.pkl'),
                                  }
RGB_features_pickle_path[4] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split4', 'features_epoch_0013_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'tsm_resnet50_nopartialbn', 'split4', 'features_epoch_0013_val_oneclip.pkl'),
                                  }

flow_features_pickle_path = {}
flow_features_pickle_path[0] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split0', 'features_epoch_0075_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split0', 'features_epoch_0075_val_oneclip.pkl'),
                                  }
flow_features_pickle_path[1] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split1', 'features_epoch_0030_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split1', 'features_epoch_0030_val_oneclip.pkl'),
                                  }
flow_features_pickle_path[2] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split2', 'features_epoch_0068_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split2', 'features_epoch_0068_val_oneclip.pkl'),
                                  }
flow_features_pickle_path[3] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split3', 'features_epoch_0044_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split3', 'features_epoch_0044_val_oneclip.pkl'),
                                  }
flow_features_pickle_path[4] = {'train': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split4', 'features_epoch_0027_traindata_testmode_oneclip.pkl'),
                                   'val': os.path.join(dataset_root, 'features', 'beoid_42verb', 'ch_epic.tsm_resnet50_flow', 'split4', 'features_epoch_0027_val_oneclip.pkl'),
                                  }

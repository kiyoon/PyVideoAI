_exec_relative_('../sparsesample_RGB_crop224_8frame_largejit_plateau_10scrop.py')


def load_model():
    return model_cfg.load_model(dataset_cfg.num_classes, input_frame_length, num_channels=3)

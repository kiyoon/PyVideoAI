_exec_relative_('../sparsesample_onehot_crop224_8frame_largejit_plateau.py')

input_type = 'gulp_flow'
num_epochs = int(os.getenv('VAI_NUM_EPOCHS', 50))
train_classifier_balanced_retraining_epochs = int(os.getenv('VAI_CRT_EPOCHS', 15))    # How many epochs to re-train the classifier from random weights, in a class-balanced way, at the end. Bingyi Kang et al. 2020, (cRT)
early_stop_patience = None

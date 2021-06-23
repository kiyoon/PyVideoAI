import argparse
import pickle
import csv
import numpy as np
import os
import sys


from sklearn.metrics import accuracy_score
from pyvideoai.utils.AP import mAP_from_AP, compute_multiple_aps
from pyvideoai.utils.multilabel_tools import count_num_samples_per_class, count_num_samples_per_class_from_csv, count_num_TP_per_class

import torch
from torch import nn
from experiment_utils.csv_to_dict import csv_to_dict
from experiment_utils.argparse_utils import add_exp_arguments
from experiment_utils import ExperimentBuilder
import dataset_configs
import model_configs
import exp_configs
import time


from pyvideoai.config import DEFAULT_EXPERIMENT_ROOT



def evaluate_pred(args):
    """evaluate accuracy"""

    dataset_cfg = dataset_configs.load_cfg(args.dataset)

    perform_multicropval=True       # when loading, assume there was multicropval. Even if there was not, having more CSV field information doesn't hurt.
    if dataset_cfg.task == 'singlelabel_classification':
        multilabel = False
        if args.mode == 'oneclip':
            best_metric_field = 'val_acc'
        else:   # multicrop
            best_metric_field = 'multi_crop_val_vid_acc_top1'
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_singlelabel(multicropval = perform_multicropval)
    elif dataset_cfg.task == 'multilabel_classification':
        multilabel = True
        if args.mode == 'oneclip':
            best_metric_field = 'val_vid_mAP'
        else:   # multicrop
            best_metric_field = 'multi_crop_val_vid_mAP'
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_multilabel(multicropval = perform_multicropval)
    else:
        raise ValueError(f"Not recognised dataset_cfg.task: {dataset_cfg.task}")

    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.model, args.experiment_name, summary_fieldnames = summary_fieldnames, summary_fieldtypes = summary_fieldtypes)
    

    if args.load_epoch == -1:
        exp.load_summary()
        load_epoch = int(exp.summary['epoch'][-1])
    elif args.load_epoch == -2:
        exp.load_summary()
        load_epoch = int(exp.get_best_model_stat(best_metric_field)['epoch'])
    elif args.load_epoch is None:
        load_epoch = None
    elif args.load_epoch >= 0:
        exp.load_summary()
        load_epoch = args.load_epoch
    else:
        raise ValueError(f"Wrong args.load_epoch value: {args.load_epoch:d}")

    if load_epoch is not None:
        predictions_file_path = os.path.join(exp.predictions_dir, f'epoch_{load_epoch:04d}_{args.mode}val.pkl')
    else:
        predictions_file_path = os.path.join(exp.predictions_dir, f'pretrained_{args.mode}val.pkl')

    with open(predictions_file_path, 'rb') as f:
        predictions = pickle.load(f)

    video_predictions = predictions['video_predictions']
    video_labels = predictions['video_labels']
    #video_ids = predictions['video_ids']

    '''
    for i in range(10):
        print(video_predictions[i].max())
        print(video_predictions[i].min())
        print(video_labels[i])
        print()
    '''

    
    if multilabel:
        APs = compute_multiple_aps(video_labels, video_predictions)
        mAP = mAP_from_AP(APs)
        print(f"mAP: {mAP:.4f}")

        num_samples_in_val = count_num_samples_per_class(video_labels)
        TPs = count_num_TP_per_class(video_labels, video_predictions)
        # !! TODO: IMPORTANT: This may not work if using the partial dataset.
        # CHANGE IT 
        train_csv = os.path.join(dataset_cfg.frames_splits_dir, dataset_cfg.split_file_basename['train'])
        num_samples_in_train = count_num_samples_per_class_from_csv(train_csv, dataset_cfg.num_classes)

        out_csv = os.path.join(exp.plots_dir, f'per_class_AP-epoch_{load_epoch:04d}_{args.mode}val.csv')
        print(f"Writing AP to {out_csv}")

        with open(out_csv, mode='w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=str(','), quotechar=str('"'), quoting=csv.QUOTE_MINIMAL)

            csvwriter.writerow(['class_key', 'AP', 'TP', 'num_samples_in_val', 'num_samples_in_train'])

            for class_key, AP, TP, class_num_samples_in_val, class_num_samples_in_train in zip(dataset_cfg.class_keys, APs, TPs, num_samples_in_val, num_samples_in_train):
                csvwriter.writerow([class_key, AP, TP, class_num_samples_in_val, class_num_samples_in_train])

    else:
        pred_labels = np.argmax(video_predictions, axis=1)
        print(accuracy_score(video_labels, pred_labels, normalize=args.normalise))


    if args.dataset == 'something_v1':
        # for Something-Something-V1, count the # of [something] in the class keys
        class_keys = dataset_cfg.class_keys 

        video_labels_kept = []
        pred_labels_kept = []
        
        for pred, label in zip(pred_labels, video_labels):
            if class_keys[label].count('[]') < 2:
                video_labels_kept.append(label)
                pred_labels_kept.append(pred)


        print(accuracy_score(video_labels_kept, pred_labels_kept, normalize=args.normalise))


        video_labels_kept = []
        pred_labels_kept = []
        
        for pred, label in zip(pred_labels, video_labels):
            if class_keys[label].count('[]') >= 2:
                video_labels_kept.append(label)
                pred_labels_kept.append(pred)


        print(accuracy_score(video_labels_kept, pred_labels_kept, normalize=args.normalise))



def main():
    parser = argparse.ArgumentParser(
        description='Load predictions, and evaluate accuracy/AP and mAP')

    add_exp_arguments(parser, dataset_configs.available_datasets, model_configs.available_models, root_default=DEFAULT_EXPERIMENT_ROOT, dataset_default='hmdb', model_default='i3d_resnet50', name_default='crop224_batch8_8x8_1scrop5tcrop_hmdbsplit1pretrained',
            dataset_channel_choices=dataset_configs.available_channels, model_channel_choices=model_configs.available_channels, exp_channel_choices=exp_configs.available_channels)
    parser.add_argument(
        '--no_normalise', action='store_false', dest='normalise', help='Do not normalise the confusion matrix')
    parser.add_argument("-l", "--load_epoch", type=int, default=None, help="Load from checkpoint. Set to -1 to load from the last checkpoint, and to -2 to load best model in terms of val_acc.")
    parser.add_argument("-m", "--mode", type=str, default="oneclip", choices=["oneclip", "multicrop"],  help="Mode used for run_val.py")

    args = parser.parse_args()

    evaluate_pred(args)



if __name__ == '__main__':
    main()


import numpy as np

def count_num_samples_per_class(video_labels):
    return np.sum(video_labels, axis=0)

def count_num_correct_preds_per_class(video_labels, video_preds):
    """TP + TN
    """
    return np.sum(np.abs(video_labels - video_preds) <= 0.5, axis=0)

def count_num_TP_per_class(video_labels, video_preds):
    """Count true positives
    """
    return np.sum((video_labels + video_preds) >= 1.5, axis=0)

def count_num_samples_per_class_from_csv(csv_path, num_classes):
    video_labels = []
    with open(csv_path, "r") as f:
        for clip_idx, path_label in enumerate(f.read().splitlines()):
            if len(path_label.split()) == 4:
                path, video_id, label, num_frames = path_label.split()
            elif len(path_label.split()) == 5:
                path, video_id, label, start, end = path_label.split()
            elif len(path_label.split()) == 1:
                continue
            else:
                raise NotImplementedError()

            label_list = label.split(",")
            label = np.zeros(num_classes, dtype=np.int)
            for label_idx in label_list:
                label[int(label_idx)] = 1       # one hot encoding
            video_labels.append(label)

    return count_num_samples_per_class(np.array(video_labels))

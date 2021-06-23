import sklearn
import numpy as np
import os

def compute_balanced_class_weight_from_csv(csv_path, label_column, num_classes, csv_separator=" ", label_separator=",", ignore_first_row=False):
    """
    params:
        csv_path (str): path to CSV
        label_column (int): column index (starting with 0) for the labels
        label_separator (str): if multilabel, separator for the labels
    """
    assert os.path.exists(csv_path), "{} not found".format(
        csv_path
    )
    y = []
    with open(csv_path, "r") as f:
        if ignore_first_row:
            f.readline()
        for line in f.read().splitlines():
            line_list = line.split(csv_separator)
            label = line_list[label_column]
            label_list = list(map(int, label.split(label_separator)))
            y.extend(label_list)
    return compute_balanced_class_weight(y, num_classes)


def compute_balanced_class_weight(y, num_classes):
    """Generate weights for loss for imbalanced dataset.
    Same as sklearn.utils.class_weight.compute_class_weight,
    but it will return "0" for the empty classes so that the length of the wieghts is `num_classes`
    """
    classes = np.unique(y)
    assert len(classes) <= num_classes, f"There are more classes than num_classes: {num_classes}"
    assert classes[-1] < num_classes, f"There are more classes ({classes[-1]+1}) than num_classes: ({num_classes})"
    weight = sklearn.utils.class_weight.compute_class_weight('balanced',
            classes = classes,
            y = y)
    if len(classes) == num_classes and classes[0] == 0 and classes[-1] == num_classes-1:
        # classes = [0, 1, 2, .., n-1]
        return weight
    else:
        extended_weight = np.zeros(num_classes, dtype=np.float)
        for class_idx, weight_val in zip(classes, weight):
            extended_weight[class_idx] = weight_val
        return extended_weight  # zero if no sample


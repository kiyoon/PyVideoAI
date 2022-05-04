# @achalddave's AP implementations
from __future__ import division

import numpy as np

def compute_average_precision(groundtruth, predictions, false_negatives=0):
    """
    Computes average precision for a binary problem. This is based off of the
    PASCAL VOC implementation.

    Args:
        groundtruth (array-like): Binary vector indicating whether each sample
            is positive or negative.
        predictions (array-like): Contains scores for each sample.
        false_negatives (int or None): In some tasks, such as object
            detection, not all groundtruth will have a corresponding prediction
            (i.e., it is not retrieved at _any_ score threshold). For these
            cases, use false_negatives to indicate the number of groundtruth
            instances that were not retrieved.

    Returns:
        Average precision.

    """
    predictions = np.asarray(predictions).squeeze()
    groundtruth = np.asarray(groundtruth, dtype=float).squeeze()

    if predictions.ndim == 0:
        predictions = predictions.reshape(-1)

    if groundtruth.ndim == 0:
        groundtruth = groundtruth.reshape(-1)

    if predictions.ndim != 1:
        raise ValueError(f'Predictions vector should be 1 dimensional, not '
                         f'{predictions.ndim}. (For multiple labels, use '
                         f'`compute_multiple_aps`.)')
    if groundtruth.ndim != 1:
        raise ValueError(f'Groundtruth vector should be 1 dimensional, not '
                         f'{groundtruth.ndim}. (For multiple labels, use '
                         f'`compute_multiple_aps`.)')

    sorted_indices = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_indices]
    groundtruth = groundtruth[sorted_indices]
    # The false positives are all the negative groundtruth instances, since we
    # assume all instances were 'retrieved'. Ideally, these will be low scoring
    # and therefore in the end of the vector.
    false_positives = 1 - groundtruth

    tp = np.cumsum(groundtruth)      # tp[i] = # of positive examples up to i
    fp = np.cumsum(false_positives)  # fp[i] = # of false positives up to i

    num_positives = tp[-1] + false_negatives

    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recalls = tp / num_positives

    # Append end points of the precision recall curve.
    precisions = np.concatenate(([0.], precisions))
    recalls = np.concatenate(([0.], recalls))

    # Find points where prediction score changes.
    prediction_changes = set(
        np.where(predictions[1:] != predictions[:-1])[0] + 1)

    num_examples = predictions.shape[0]

    # Recall and scores always "change" at the first and last prediction.
    c = prediction_changes | set([0, num_examples])
    c = np.array(sorted(list(c)), dtype=np.int)

    precisions = precisions[c[1:]]

    # Set precisions[i] = max(precisions[j] for j >= i)
    # This is because (for j > i), recall[j] >= recall[i], so we can always use
    # a lower threshold to get the higher recall and higher precision at j.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    ap = np.sum((recalls[c[1:]] - recalls[c[:-1]]) * precisions)

    return ap


def compute_multiple_aps(groundtruth, predictions, false_negatives=None):
    """Convenience function to compute APs for multiple labels.

    Args:
        groundtruth (np.array): Shape (num_samples, num_labels)
        predictions (np.array): Shape (num_samples, num_labels)
        false_negatives (list or None): In some tasks, such as object
            detection, not all groundtruth will have a corresponding prediction
            (i.e., it is not retrieved at _any_ score threshold). For these
            cases, use false_negatives to indicate the number of groundtruth
            instances which were not retrieved for each category.

    Returns:
        aps_per_label (np.array, shape (num_labels,)): Contains APs for each
            label. NOTE: If a label does not have positive samples in the
            groundtruth, the AP is set to -1.
    """
    predictions = np.asarray(predictions)
    groundtruth = np.asarray(groundtruth)
    if predictions.ndim != 2:
        raise ValueError('Predictions should be 2-dimensional,'
                         ' but has shape %s' % (predictions.shape, ))
    if groundtruth.ndim != 2:
        raise ValueError('Groundtruth should be 2-dimensional,'
                         ' but has shape %s' % (predictions.shape, ))

    num_labels = groundtruth.shape[1]
    aps = np.zeros(groundtruth.shape[1])
    if false_negatives is None:
        false_negatives = [0] * num_labels
    for i in range(num_labels):
        if not groundtruth[:, i].any():
#            print('WARNING: No groundtruth for label: %s' % i)
            aps[i] = -1
        else:
            aps[i] = compute_average_precision(groundtruth[:, i],
                                               predictions[:, i],
                                               false_negatives[i])
    return aps


# CATER implementation
# Kiyoon addition: only include if many samples.
def mAP_from_AP(AP, class_include_mask = None):
    if class_include_mask is None:
        mAP = np.mean([el for el in AP if el >= 0])
    else:
        assert len(AP) == len(class_include_mask)
        mAP = np.mean([el for el, include in zip(AP, class_include_mask) if include and el >= 0])

    return mAP


def get_classes_to_include_mask(labels, exclude_classes_less_sample_than = 1):
    assert exclude_classes_less_sample_than >= 1
    num_classes = labels.shape[1]
    num_samples = [0] * num_classes

    for label in labels:
        for idx, gt in enumerate(label):
            if gt == 1:
                num_samples[idx] += 1
            elif gt == 0:
                pass
            else:
                raise ValueError(f'label include not 0 or 1 value: {gt}')

    return [num_sample >= exclude_classes_less_sample_than for num_sample in num_samples]


def mAP(labels, preds, exclude_classes_less_sample_than = 1):
    assert exclude_classes_less_sample_than >= 1
    if exclude_classes_less_sample_than > 1:
        class_include_mask = get_classes_to_include_mask(labels, exclude_classes_less_sample_than)
    else:
        class_include_mask = None

    AP = compute_multiple_aps(labels, preds)
    return mAP_from_AP(AP, class_include_mask)

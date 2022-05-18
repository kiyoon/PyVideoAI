from __future__ import annotations
import numpy as np
import scipy.spatial as spatial
import time

import logging
logger = logging.getLogger(__name__)


def find_distance_by_average_num_neighbours(features: np.array, target_num_neighbours: int) -> float:
    distance = 1.
    neighbour_indices = feature_neighbours_within_distance(features, distance)
    num_neighbours = [len(indices) for indices in neighbour_indices]
    print(sum(num_neighbours) / len(num_neighbours))


def feature_neighbours_within_distance(features: np.array, distance: float) -> list[list[int]]:
    logger.info(f'Getting neighbours within distance {distance:.4f}')
    start_time = time.time()

    feature_tree = spatial.cKDTree(features)
    #neighbour_indices = []
    neighbour_indices = feature_tree.query_ball_tree(feature_tree, distance)

    # Remove current sample as a neighbour. Otherwise it will include itself.
    for idx, neighbours in enumerate(neighbour_indices):
        neighbours.remove(idx)

    elapsed_time = time.time() - start_time
    logger.info(f'Getting features took {elapsed_time:.1f} seconds')
    return neighbour_indices
#    for feature_idx, feature in tqdm.tqdm(enumerate(features), total=len(features)):
#        neighbour_idx: list[int] = list(feature_tree.query_ball_point(feature, distance))
#        neighbour_idx.remove(feature_idx)   # exclude current sample as a neighbour.
#        neighbour_indices.append(neighbour_idx)
#
#        num_neighbours = [len(indices) for indices in neighbour_indices]
#        print(sum(num_neighbours) / len(num_neighbours))


def main():
    import pickle
    import dataset_configs
    logging.basicConfig(level='INFO')
    dataset_cfg = dataset_configs.load_cfg('epic100_verb_features', 'beta')

    split = 'train'
    input_feature_dim = 2048 * 2

    RGB_feature_pickle_path = dataset_cfg.RGB_features_pickle_path[split]
    flow_feature_pickle_path = dataset_cfg.flow_features_pickle_path[split]

    with open(RGB_feature_pickle_path, 'rb') as f:
        d = pickle.load(f)

    RGB_video_ids, RGB_labels, RGB_features = d['video_ids'], d['labels'], d['clip_features']
    if RGB_features.shape[1] != 2048:
        # features are not averaged. Average now
        RGB_features = np.mean(RGB_features, axis=1)

    with open(flow_feature_pickle_path, 'rb') as f:
        d = pickle.load(f)

    flow_video_ids, flow_labels, flow_features = d['video_ids'], d['labels'], d['clip_features']
    assert RGB_video_ids.shape[0] == RGB_labels.shape[0] == RGB_features.shape[0] == flow_video_ids.shape[0] == flow_labels.shape[0] == flow_features.shape[0]

    if flow_features.shape[1] != 2048:
        # features are not averaged. Average now
        flow_features = np.mean(flow_features, axis=1)

    # Concatenate RGB and flow features.
    flow_video_id_to_idx = {}
    for idx, video_id in enumerate(flow_video_ids):
        assert video_id not in flow_video_id_to_idx
        flow_video_id_to_idx[video_id] = idx

    # Loop over RGB features and find flow features. The array ordering may be different.
    concat_RGB_flow_features = np.zeros((RGB_features.shape[0], input_feature_dim), dtype=RGB_features.dtype)
    for idx, (video_id, RGB_feature) in enumerate(zip(RGB_video_ids, RGB_features)):
        flow_idx = flow_video_id_to_idx[video_id]
        concat_feature = np.concatenate((RGB_feature, flow_features[flow_idx]))
        concat_RGB_flow_features[idx] = concat_feature

    video_ids, labels, features = RGB_video_ids, RGB_labels, concat_RGB_flow_features
    find_distance_by_average_num_neighbours(features, 10)


if __name__ == '__main__':
    main()

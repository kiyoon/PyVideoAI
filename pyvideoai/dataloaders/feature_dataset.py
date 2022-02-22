#!/usr/bin/env python3

import numpy as np
import os
import random
import torch
import torch.utils.data

import logging

logger = logging.getLogger(__name__)

class FeatureDataset(torch.utils.data.Dataset):
    """
    Feature loader.
    """

    def __init__(self, video_ids: np.array, labels: np.array, features: np.array):
        logger.info("Constructing feature dataset...")
        assert video_ids.shape[0] == labels.shape[0] == features.shape[0], ('Incorrect shape of the inputs. Make sure video_ids, labels and features have the same number of inputs.\n'
                                                                            f'Got {video_ids.shape[0] = }, {labels.shape[0] = }, {features.shape[0] = }')
        assert video_ids.ndim == 1
        assert labels.ndim in [1, 2]
        # features can be any dimension

        logger.info(f'Constructed feature dataloader with size {video_ids.shape[0]}')

        self._video_ids = video_ids
        self._labels = labels
        self._features = features

        

    def filter_samples(self, video_ids: list):
        """Given a video_ids list, filter the samples.
        Used for visualisation.
        """
        indices_of_video_ids = [x for x, v in enumerate(self._video_ids) if v in video_ids]

        self._video_ids = self._video_ids[indices_of_video_ids]
        self._labels = self._labels[indices_of_video_ids]
        self._features = self._features[indices_of_video_ids]


    def __getitem__(self, index):
        return self._features[index], self._video_ids[index], self._labels[index], index

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._video_ids)


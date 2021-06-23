# Deprecated!
# alias for FramesSparsesampleDataset

import logging
logger = logging.getLogger(__name__)
logger.warning('Deprecated: FramesSparsesampleDatasetAdvanced is now renamed to FramesSparsesampleDataset and it will load the latter.')
from .frames_sparsesample_dataset import *
from .frames_sparsesample_dataset import FramesSparsesampleDataset as FramesSparsesampleDatasetAdvanced


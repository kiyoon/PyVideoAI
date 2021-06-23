import torch
import numpy as np
from torch import nn

import logging

logger = logging.getLogger(__name__)

class feature_model(nn.Module):
    def __init__(self, input_feature_dim, num_layers=2, num_units=64, num_classes = 36):
        super(feature_model, self).__init__()

        self.input_feature_dim = input_feature_dim
        self.block_dict = nn.ModuleDict()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_classes = num_classes
        self.build_block()

    def build_block(self):
        logger.info("Assuming input is of size (b=4, input_feature_dim=%d)" % self.input_feature_dim)
        out = torch.zeros((4, self.input_feature_dim))

        for idx_layer in range(self.num_layers):
            self.block_dict['g_fcc_{}'.format(idx_layer)] = nn.Linear(out.shape[1], out_features=self.num_units, bias=True)
            out = self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out)
            self.block_dict['LeakyReLU_{}'.format(idx_layer)] = nn.LeakyReLU()
            out = self.block_dict['LeakyReLU_{}'.format(idx_layer)].forward(out)
            self.block_dict['BN_{}'.format(idx_layer)] = nn.BatchNorm1d(out.shape[1])
            out = self.block_dict['BN_{}'.format(idx_layer)].forward(out)

        if self.num_classes > 0:
            self.block_dict['classify_fc_{}'.format(idx_layer)] = nn.Linear(out.shape[1], out_features=self.num_classes, bias=True)
            out = self.block_dict['classify_fc_{}'.format(idx_layer)].forward(out)

        logger.info('Block built with output volume shape: %s', out.shape)

    def forward(self, x):
        for idx_layer in range(self.num_layers):
            x = self.block_dict['g_fcc_{}'.format(idx_layer)].forward(x)
            x = self.block_dict['LeakyReLU_{}'.format(idx_layer)].forward(x)
            x = self.block_dict['BN_{}'.format(idx_layer)].forward(x)

        if self.num_classes > 0:
            x = self.block_dict['classify_fc_{}'.format(idx_layer)].forward(x)

        return x

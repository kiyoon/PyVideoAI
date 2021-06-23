import torch
from torch import nn

import numpy as np

import logging
logger = logging.getLogger(__name__)

class BatchRelationalModule(nn.Module):
    def __init__(self, input_shape, use_coordinates=True, num_layers=2, num_units=64):
        super(BatchRelationalModule, self).__init__()

        self.input_shape = input_shape      # (F,H,W) or (F,H*W)
        self.block_dict = nn.ModuleDict()
        self.first_time = True
        self.use_coordinates = use_coordinates
        self.num_layers = num_layers
        self.num_units = num_units
        self.build_block()

    def build_block(self):
        logger.info("Building relational module of input shape: {:s}".format(str(self.input_shape)))
        logger.info("Assuming the batch size is 1")
        self.input_shape = (1,) + self.input_shape


        out_img = torch.zeros(self.input_shape)

        """g"""
        if len(out_img.shape) > 3:
            b, c, h, w = out_img.shape
            out_img = out_img.view(b, c, h * w)

        out_img = out_img.permute([0, 2, 1])  # h*w, c
        b, length, c = out_img.shape
        logger.debug(out_img.shape)
        # x_flat = (64 x 25 x 24)
        if self.use_coordinates:
            self.coord_tensor = []
            for i in range(length):
                self.coord_tensor.append(torch.Tensor(np.array([i])))

            self.coord_tensor = torch.stack(self.coord_tensor, dim=0).unsqueeze(0)

            if self.coord_tensor.shape[0] != out_img.shape[0]:
                self.coord_tensor = self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])

            out_img = torch.cat([out_img, self.coord_tensor], dim=2)

        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)
        x_i = x_i.repeat(1, length, 1, 1)  # (h*wxh*wxc)
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc)
        x_j = x_j.repeat(1, 1, length, 1)  # (h*wxh*wxc)

        # concatenate all together
        per_location_feature = torch.cat([x_i, x_j], 3)  # (h*wxh*wx2*c)

        out = per_location_feature.view(
            per_location_feature.shape[0] * per_location_feature.shape[1] * per_location_feature.shape[2],
            per_location_feature.shape[3])
        logger.debug(out.shape)
        for idx_layer in range(self.num_layers):
            self.block_dict['g_fcc_{}'.format(idx_layer)] = nn.Linear(out.shape[1], out_features=self.num_units, bias=True)
            out = self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out)
            self.block_dict['LeakyReLU_{}'.format(idx_layer)] = nn.LeakyReLU()
            out = self.block_dict['LeakyReLU_{}'.format(idx_layer)].forward(out)

        # reshape again and sum
        logger.debug(out.shape)
        out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
        logger.debug(out.shape)
        out = out.sum(1).sum(1)
        logger.debug('here', out.shape)
        """f"""
        self.post_processing_layer = nn.Linear(in_features=out.shape[1], out_features=self.num_units)
        out = self.post_processing_layer.forward(out)
        self.block_dict['LeakyReLU_post_processing'] = nn.LeakyReLU()
        out = self.block_dict['LeakyReLU_post_processing'].forward(out)
        self.output_layer = nn.Linear(in_features=out.shape[1], out_features=self.num_units)
        out = self.output_layer.forward(out)
        self.block_dict['LeakyReLU_output'] = nn.LeakyReLU()
        out = self.block_dict['LeakyReLU_output'].forward(out)
        logger.info('Block built with output volume shape {:s}'.format(str(out.shape)))

    def forward(self, x_img):

        out_img = x_img
        # print("input", out_img.shape)
        """g"""
        if len(out_img.shape) > 3:
            b, c, h, w = out_img.shape
            out_img = out_img.view(b, c, h * w)

        out_img = out_img.permute([0, 2, 1])  # h*w, c
        b, length, c = out_img.shape

        if self.use_coordinates:
            if self.coord_tensor.shape[0] != out_img.shape[0]:
                self.coord_tensor = self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])

            out_img = torch.cat([out_img, self.coord_tensor.to(x_img.device)], dim=2)
        # x_flat = (64 x 25 x 24)
        # print('out_img', out_img.shape)
        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)
        x_i = x_i.repeat(1, length, 1, 1)  # (h*wxh*wxc)
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc)
        x_j = x_j.repeat(1, 1, length, 1)  # (h*wxh*wxc)

        # concatenate all together
        per_location_feature = torch.cat([x_i, x_j], 3)  # (h*wxh*wx2*c)
        out = per_location_feature.view(
            per_location_feature.shape[0] * per_location_feature.shape[1] * per_location_feature.shape[2],
            per_location_feature.shape[3])

        for idx_layer in range(2):
            out = self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out)
            out = self.block_dict['LeakyReLU_{}'.format(idx_layer)].forward(out)

        # reshape again and sum
        # print(out.shape)
        out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
        out = out.sum(1).sum(1)

        """f"""
        out = self.post_processing_layer.forward(out)
        out = self.block_dict['LeakyReLU_post_processing'].forward(out)
        out = self.output_layer.forward(out)
        out = self.block_dict['LeakyReLU_output'].forward(out)
        # print('Block built with output volume shape', out.shape)
        return out

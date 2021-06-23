import torch
import numpy as np
from torch import nn

class BatchRelationalModule(nn.Module):
    def __init__(self, input_feature_dim, use_coordinates=False, num_layers=2, num_units=64):
        super(BatchRelationalModule, self).__init__()

        self.input_feature_dim = input_feature_dim
        self.block_dict = nn.ModuleDict()
        self.first_time = True
        self.use_coordinates = use_coordinates
        self.num_layers = num_layers
        self.num_units = num_units
        self.build_block()

    def build_block(self):
        print("Assuming input is of size (b=4, num_object=4, input_feature_dim=%d)" % self.input_feature_dim)
        out_img = torch.zeros((4, 4, self.input_feature_dim))
        """g"""
        b, length, c = out_img.shape
        print(out_img.shape)
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
        print(out.shape)
        for idx_layer in range(self.num_layers):
            self.block_dict['g_fcc_{}'.format(idx_layer)] = nn.Linear(out.shape[1], out_features=self.num_units, bias=True)
            out = self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out)
            self.block_dict['LeakyReLU_{}'.format(idx_layer)] = nn.LeakyReLU()
            out = self.block_dict['LeakyReLU_{}'.format(idx_layer)].forward(out)

        # reshape again and sum
        print(out.shape)
        out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
        out = out.sum(1).sum(1)
        print('here', out.shape)
        """f"""
        self.post_processing_layer = nn.Linear(in_features=out.shape[1], out_features=self.num_units)
        out = self.post_processing_layer.forward(out)
        self.block_dict['LeakyReLU_post_processing'] = nn.LeakyReLU()
        out = self.block_dict['LeakyReLU_post_processing'].forward(out)
        self.output_layer = nn.Linear(in_features=out.shape[1], out_features=self.num_units)
        out = self.output_layer.forward(out)
        self.block_dict['LeakyReLU_output'] = nn.LeakyReLU()
        out = self.block_dict['LeakyReLU_output'].forward(out)
        print('Block built with output volume shape', out.shape)

    def forward(self, x_img):
        if isinstance(x_img, list):
            # variable length feature count

            batch_size = len(x_img)
            length_per_batch = []
            batch_to_g_size = 0     # size of batch that will go into the g network
            for x_img1 in x_img:
                length, c = x_img1.shape
                length_per_batch.append(length)
                batch_to_g_size += length

            out = torch.Tensor()
            out_list = [None] * batch_size
            for b, x_img1 in enumerate(x_img):
                out_img = x_img1
                """g"""
                length, c = out_img.shape

                if self.use_coordinates:
                    if self.coord_tensor.shape[0] != out_img.shape[0]:
                        self.coord_tensor = self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])

                    #print(self.coord_tensor)
                    out_img = torch.cat([out_img, self.coord_tensor.to(x_img1.device)], dim=2)
                # x_flat = (64 x 25 x 24)
                # print('out_img', out_img.shape)
                x_i = torch.unsqueeze(out_img, 0)  # (1xh*wxc)
                x_i = x_i.repeat(length, 1, 1)  # (h*wxh*wxc)
                x_j = torch.unsqueeze(out_img, 1)  # (h*wx1xc)
                x_j = x_j.repeat(1, length, 1)  # (h*wxh*wxc)

                # concatenate all together
                per_location_feature = torch.cat([x_i, x_j], 2)  # (h*wxh*wx2*c)

                out_list[b] = per_location_feature.view(
                    per_location_feature.shape[0] * per_location_feature.shape[1],
                    per_location_feature.shape[2])

            out = torch.cat(out_list)

            #print(out.shape)

            for idx_layer in range(self.num_layers):
                out = self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out)
                out = self.block_dict['LeakyReLU_{}'.format(idx_layer)].forward(out)

            #print(out.shape)

            # reshape again and sum
            out_idx = 0
            #out_list = [None] * batch_size
            for b in range(batch_size):

                out_list[b] = out[out_idx:out_idx+length_per_batch[b] ** 2].view(length_per_batch[b], length_per_batch[b], -1)
                out_list[b] = out_list[b].sum(0).sum(0).unsqueeze(0)
                out_idx += length_per_batch[b] ** 2

            out = torch.cat(out_list, 0)
            #print(out.shape)

            """f"""
            out = self.post_processing_layer.forward(out)
            out = self.block_dict['LeakyReLU_post_processing'].forward(out)
            out = self.output_layer.forward(out)
            out = self.block_dict['LeakyReLU_output'].forward(out)
            # print('Block built with output volume shape', out.shape)
            return out


        else:
            # constant feature count
            out_img = x_img
            """g"""
            b, length, c = out_img.shape

            if self.use_coordinates:
                if self.coord_tensor.shape[0] != out_img.shape[0]:
                    self.coord_tensor = self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])

                #print(self.coord_tensor)
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

            for idx_layer in range(self.num_layers):
                out = self.block_dict['g_fcc_{}'.format(idx_layer)].forward(out)
                out = self.block_dict['LeakyReLU_{}'.format(idx_layer)].forward(out)

            # reshape again and sum
            # print(out.shape)
            out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
            #out = out.sum(1).sum(1)
            out = out.mean(1).mean(1)

            """f"""
            out = self.post_processing_layer.forward(out)
            out = self.block_dict['LeakyReLU_post_processing'].forward(out)
            out = self.output_layer.forward(out)
            out = self.block_dict['LeakyReLU_output'].forward(out)
            # print('Block built with output volume shape', out.shape)
            return out

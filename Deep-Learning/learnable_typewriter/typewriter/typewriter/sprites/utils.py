import numpy as np
import torch
import omegaconf
from torch import nn, stack, randn, from_numpy, full, zeros
from scipy import signal

def create_mlp_with_conv1d(in_ch, out_ch, n_hidden_units, n_layers, norm_layer=None):
    if norm_layer is None or norm_layer in ['id', 'identity']:
        norm_layer = nn.Identity
    elif norm_layer in ['batch_norm', 'bn']:
        norm_layer = nn.BatchNorm1d
    elif not norm_layer == nn.BatchNorm1d:
        raise NotImplementedError

    if n_layers > 0:
        seq = [nn.Conv1d(in_ch, n_hidden_units, kernel_size=1), norm_layer(n_hidden_units), nn.ReLU(True)]
        for _ in range(n_layers - 1):
            seq += [nn.Conv1d(n_hidden_units, n_hidden_units, kernel_size=1), nn.ReLU(True)]
        seq += [nn.Conv1d(n_hidden_units, out_ch, kernel_size=1)]
    else:
        seq = [nn.Conv1d(in_ch, out_ch, kernel_size=1)]
    return nn.Sequential(*seq)

def create_gaussian_weights(img_size, n_channels, std=6):
    g1d_h = signal.gaussian(img_size[0], std)
    g1d_w = signal.gaussian(img_size[1], std)
    g2d = np.outer(g1d_h, g1d_w)
    return from_numpy(g2d).unsqueeze(0).repeat(n_channels, 1, 1).float()

def init_objects(K, c, size, init):
    samples = []
    for _ in range(K):
        if 'constant' in init:
            cons = init['constant']
            if isinstance(cons, omegaconf.listconfig.ListConfig):
                sample = zeros((c, ) + tuple(size), dtype=torch.float)
                for c_id in range(c):
                    sample[c_id, ...] = cons[c_id]
                # sample[]
            else:
                sample = full((c, ) + tuple(size), cons, dtype=torch.float)
        elif 'gaussian' in init:
            sample = create_gaussian_weights(size, c, init['gaussian'])
        else:
            raise NotImplementedError
        samples.append(sample)

    return stack(samples)

def copy_with_noise(t, noise_scale=0.0001):
    return t.detach().clone() + randn(t.shape, device=t.device) * noise_scale


class TPSGrid(nn.Module):
    """Original implem: https://github.com/WarBean/tps_stn_pytorch"""

    def __init__(self, img_size, target_control_points):
        super().__init__()
        img_height, img_width = img_size
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = img_height * img_width
        y, x = torch.meshgrid(torch.linspace(-1, 1, img_height), torch.linspace(-1, 1, img_width))
        target_coordinate = torch.stack([x.flatten(), y.flatten()], 1)
        target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate], 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    @staticmethod
    def compute_partial_repr(input_points, control_points):
        """Compute radial basis kernel phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2"""
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
        repr_matrix.masked_fill_(repr_matrix != repr_matrix, 0)
        return repr_matrix

    def forward(self, source_control_points):
        Y = torch.cat([source_control_points, self.padding_matrix.expand(source_control_points.size(0), 3, 2)], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)
        return source_coordinate
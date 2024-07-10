# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
import torchvision
# from spikingjelly.activation_based import neuron, functional, surrogate, layer

from torch.nn.functional import kl_div
from torch.autograd import Variable



def tie_weights(src, trg):
    assert type(src) == type(trg)
    try:
        trg.weight = src.weight
        trg.bias = src.bias
    except:
        trg = src

def preprocess_obs(rgb_obs, dvs_obs, dvs_obs_shape):
    # print("raw dvs_obs:", torch.isnan(dvs_obs).all(), dvs_obs.min(), dvs_obs.max())

    # RGB
    rgb_obs = rgb_obs / 255.

    # DVS
    if dvs_obs_shape[0] == 5 * 3:
        nonzero_ev = (dvs_obs != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            # compute mean and stddev of the **nonzero** elements of the event tensor
            # we do not use PyTorch's default mean() and std() functions since it's faster
            # to compute it by hand than applying those funcs to a masked array
            mean = dvs_obs.sum() / num_nonzeros
            stddev = torch.sqrt((dvs_obs ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero_ev.float()
            if stddev != 0:
                dvs_obs = mask * (dvs_obs - mean) / stddev
        # pass
    elif dvs_obs_shape[0] == 2 * 3:
        dvs_obs = dvs_obs / 255.

    # print("after dvs_obs:", torch.isnan(dvs_obs).all(), dvs_obs.min(), dvs_obs.max())

    return rgb_obs, dvs_obs


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.obs_shape = obs_shape

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = 6
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):

        if self.obs_shape[0] != 5:
            # ↓↓↓ RGB，DVS-frame，E2VID preprocess
            obs = obs / 255.
            # ↑↑↑

        else:
            # ↓↓↓ DVS-Voxel-grid preprocess！！！
            nonzero_ev = (obs != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                # compute mean and stddev of the **nonzero** elements of the event tensor
                # we do not use PyTorch's default mean() and std() functions since it's faster
                # to compute it by hand than applying those funcs to a masked array
                mean = obs.sum() / num_nonzeros
                stddev = torch.sqrt((obs ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                obs = mask * (obs - mean) / stddev
            # ↑↑↑

        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out

        return out, None

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderCarla096(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))

        out_dims = 4*4  # if defaults change, adjust this as needed
        self.fc = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class PixelEncoderCarla098(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.obs_shape = obs_shape

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        out_dims = 6*6  # 1 cameras, input: 128*128
        # out_dims = 14*14  # 1 cameras, input: 256*256

        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()



class pixelCat(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super().__init__()

        assert len(obs_shape) == 2

        rgb_obs_shape, dvs_obs_shape = obs_shape[0], obs_shape[1]

        self.rgb_obs_shape = rgb_obs_shape
        self.dvs_obs_shape = dvs_obs_shape

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        out_dims = 36  # 3 cameras

        self.rgb_head_convs = nn.ModuleList()
        self.rgb_head_convs.append(nn.Conv2d(3 * 3, 64, 5, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.rgb_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.rgb_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.rgb_ln = nn.LayerNorm(self.feature_dim)

        self.dvs_head_convs = nn.ModuleList()
        self.dvs_head_convs.append(nn.Conv2d(5 * 3, 64, 5, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.dvs_head_convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.dvs_fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.dvs_ln = nn.LayerNorm(self.feature_dim)

        # out_dims = 16  # 1 cameras

        self.last_fusion_fc = nn.Linear(self.feature_dim * 2, self.feature_dim)
        self.last_fusion_ln = nn.LayerNorm(self.feature_dim)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward_conv(self, obs):
        rgb_obs, dvs_obs = obs
        # print("rgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("dvs_obs:", dvs_obs.min(), dvs_obs.max())

        # Obs Preprocess ↓
        # RGB的预处理：
        rgb_obs, dvs_obs = preprocess_obs(rgb_obs, dvs_obs, self.dvs_obs_shape)
        # ↑↑↑
        # print("ffffrgb_obs:", rgb_obs.min(), rgb_obs.max())
        # print("ffffdvs_obs:", dvs_obs.min(), dvs_obs.max())

        # print(torch.min(rgb_obs), torch.max(rgb_obs), torch.min(dvs_obs), torch.max(dvs_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[0](rgb_obs))
        rgb_conv = torch.relu(self.rgb_head_convs[1](rgb_conv))
        rgb_conv = torch.relu(self.rgb_head_convs[2](rgb_conv))
        rgb_conv = torch.relu(self.rgb_head_convs[3](rgb_conv))

        dvs_conv = torch.relu(self.dvs_head_convs[0](dvs_obs))
        dvs_conv = torch.relu(self.dvs_head_convs[1](dvs_conv))
        dvs_conv = torch.relu(self.dvs_head_convs[2](dvs_conv))
        dvs_conv = torch.relu(self.dvs_head_convs[3](dvs_conv))

        return rgb_conv, dvs_conv


    def forward(self, obs, detach=False):
        rgb_conv, dvs_conv = self.forward_conv(obs)

        if detach:
            rgb_conv = rgb_conv.detach()
            dvs_conv = dvs_conv.detach()

        rgb_h = rgb_conv.view(rgb_conv.size(0), -1)
        rgb_h = self.rgb_ln(self.rgb_fc(rgb_h))

        dvs_h = dvs_conv.view(dvs_conv.size(0), -1)
        dvs_h = self.dvs_ln(self.dvs_fc(dvs_h))

        return torch.cat([rgb_h, dvs_h], dim=1), [rgb_h, dvs_h]

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(4):
            tie_weights(src=source.rgb_head_convs[i], trg=self.rgb_head_convs[i])
            tie_weights(src=source.dvs_head_convs[i], trg=self.dvs_head_convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            # L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)
        #
        # for i in range(self.num_layers):
        #     L.log_param('train_encoder/rgb_conv%s' % (i + 1), self.rgb_convs[i], step)
        #     L.log_param('train_encoder/dvs_conv%s' % (i + 1), self.dvs_convs[i], step)
        # L.log_param('train_encoder/fc', self.fc, step)
        # L.log_param('train_encoder/ln', self.ln, step)










_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixelCarla096': PixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                       'pixelCat': pixelCat,
                       }


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, stride
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, stride
    )

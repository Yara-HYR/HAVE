# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

class ChannelAttention(nn.Module):
    """
    The implementation of channel attention mechanism.
    """

    def __init__(self, channel, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SpatialAttention(nn.Module):
    """
    The implementation of spatial attention mechanism.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        weight_map = self.sigmoid(x)
        return weight_map


class Attention(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 4,
            kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 4,
            kernel_size=1
        )

        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 4,
            kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )

        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

    def forward(self, x1,x2,x3,xt):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x1.size()
        proj_query = self\
            .query_conv(x3)\
            .view(m_batchsize, -1, width * height)\
            .permute(0, 2, 1)
        proj_key = self\
            .key_conv(x1)\
            .view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)







        proj_query2 = self\
            .query_conv(x3)\
            .view(m_batchsize, -1, width * height)\
            .permute(0, 2, 1)
        proj_key2 = self\
            .key_conv2(x2)\
            .view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax2(energy2)

        proj_value = self\
            .value_conv(x3)\
            .view(m_batchsize, -1, width * height)

        attention = self.softmax(attention+attention2)

        out = torch.bmm(
            proj_value,
            attention.permute(0, 2, 1)
        )
        attention_mask = out.view(m_batchsize, C, height, width)

  
        proj_value_t = self\
            .value_conv2(xt)\
            .view(m_batchsize, -1, width * height)

        out2 = torch.bmm(
            proj_value_t,
            attention.permute(0, 2, 1)
        )
        attention_mask2 = out2.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask + x3
        out2 = self.gamma2 * attention_mask2 + xt
        return attention,out,out2








class SelfAttention(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 4,
            kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 4,
            kernel_size=1
        )

        self.value_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )



        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)



    def forward(self, xi):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = xi.size()
        proj_query = self\
            .query_conv(xi)\
            .view(m_batchsize, -1, width * height)\
            .permute(0, 2, 1)
        proj_key = self\
            .key_conv(xi)\
            .view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)



        proj_value = self\
            .value_conv(xi)\
            .view(m_batchsize, -1, width * height)

        # attention = attention+attention2

        out = torch.bmm(
            proj_value,
            attention.permute(0, 2, 1)
        )
        attention_mask = out.view(m_batchsize, C, height, width)

        attention = self.gamma * attention_mask + xi


        return attention




class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        # h = conv.view(conv.size(0), -1)
        return conv



    def forward_conv1(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs1[0](obs))
        self.outputs1['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs1[i](conv))
            self.outputs1['conv%s' % (i + 1)] = conv

        # h = conv.view(conv.size(0), -1)
        return conv





    def forward(self, obs, detach=False,fusion=True):
        diff  = torch.cat([obs[:,6:9,:,:]-obs[:,3:6,:,:],obs[:,3:6,:,:]-obs[:,0:3,:,:]],dim=1)
        h = self.forward_conv(diff)
        h1 = self.forward_conv1(obs[:,3:6,:,:])

        h0 = self.forward_conv1(obs[:,0:3,:,:])
        h2 = self.forward_conv1(obs[:,6:9,:,:])#
        # import pdb
        # pdb.set_trace()

        # h1 = self.selfatt_image(h1)
        # h = self.selfatt_video(h)

        s_att, h1,h = self.att_qkv(h0,h2,h1,h)
        # # h = s_att*h + h




        # h = self.crossatt_qkvv(h,h1) + self.selfatt_qkvv(h,h)
        # h1 = self.crossatt_qkvi(h1,h) + self.selfatt_qkvi(h1,h1)
        


        # i_attention = self.spatial_attention(h1)
        # # i_attention2 =self.channel_attention(h1)

        # v_attention = self.temporal_attention(h)
        # # v_attention2 =self.channel_attention2(h)

        # h = h*i_attention + h* v_attention + h*i_attention2 +h*v_attention2+h
        # h1 = h1*v_attention + h1*i_attention + h1*i_attention2 + h1*v_attention2+h1

        # h = h*i_attention + h* v_attention + h
        # h1 = h1*v_attention + h1*i_attention + h1

        h = h.view(h.size(0), -1)
        h1 = h1.view(h1.size(0), -1)

        if detach:
            h = h.detach()
            h1 = h1.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out



        h_fc1 = self.fc1(h1)
        self.outputs1['fc'] = h_fc1

        out1 = self.ln(h_fc1)
        self.outputs1['ln'] = out1

        # fusion_out = torch.cat([out,out1],dim=1)


        fusion_out = self.blp(h_fc,h_fc1)
        fusion_out = self.ln_a(fusion_out)

        if fusion==True:
            return fusion_out
        else:
            return fusion_out, out, out1

    # def forward(self, obs, detach=False,fusion=True):
    #     h = self.forward_conv(obs)
    #     h1 = self.forward_conv1(obs[:,3:6,:,:])

    #     h0 = self.forward_conv1(obs[:,0:3,:,:])
    #     h2 = self.forward_conv1(obs[:,6:9,:,:])#
    #     # import pdb
    #     # pdb.set_trace()

    #     # s_att, h1,h = self.att_qkv(h0,h2,h1,h)


    #     h = self.crossatt_qkvv(h,h1) + self.selfatt_qkvv(h,h)
    #     h1 = self.crossatt_qkvi(h1,h) + self.selfatt_qkvi(h1,h1)
    #     # h = s_att*h + h


    #     # i_attention = self.spatial_attention(h1)
    #     # # i_attention2 =self.channel_attention(h1)

    #     # v_attention = self.temporal_attention(h)
    #     # # v_attention2 =self.channel_attention2(h)

    #     # h = h*i_attention + h* v_attention + h*i_attention2 +h*v_attention2+h
    #     # h1 = h1*v_attention + h1*i_attention + h1*i_attention2 + h1*v_attention2+h1

    #     # h = h*i_attention + h* v_attention + h
    #     # h1 = h1*v_attention + h1*i_attention + h1

    #     h = h.view(h.size(0), -1)
    #     h1 = h1.view(h1.size(0), -1)

    #     if detach:
    #         h = h.detach()
    #         h1 = h1.detach()

    #     h_fc = self.fc(h)
    #     self.outputs['fc'] = h_fc

    #     out = self.ln(h_fc)
    #     self.outputs['ln'] = out



    #     h_fc1 = self.fc1(h1)
    #     self.outputs1['fc'] = h_fc1

    #     out1 = self.ln(h_fc1)
    #     self.outputs1['ln'] = out1

    #     fusion_out = torch.cat([out,out1],dim=1)
    #     if fusion==True:
    #         return fusion_out
    #     else:
    #         return fusion_out, out, out1


    # def forward_conv(self, obs):
    #     obs = obs / 255.
    #     self.outputs['obs'] = obs

    #     convs_list = []

    #     conv = torch.relu(self.convs[0](obs))
    #     self.outputs['conv1'] = conv
    #     convs_list.append(conv)

    #     for i in range(1, self.num_layers):
    #         conv = torch.relu(self.convs[i](conv))
    #         self.outputs['conv%s' % (i + 1)] = conv
    #         convs_list.append(conv)

    #     # h = conv.view(conv.size(0), -1)
    #     return convs_list[0],convs_list[1],convs_list[2],convs_list[3]
    #     # return conv



    # def forward_conv1(self, obs):
    #     obs = obs / 255.
    #     self.outputs['obs'] = obs

    #     convs_list = []

    #     conv = torch.relu(self.convs1[0](obs))
    #     self.outputs1['conv1'] = conv

    #     convs_list.append(conv)

    #     for i in range(1, self.num_layers):
    #         conv = torch.relu(self.convs1[i](conv))
    #         self.outputs1['conv%s' % (i + 1)] = conv
    #         convs_list.append(conv)


    #     # h = conv.view(conv.size(0), -1)
    #     return convs_list[0],convs_list[1],convs_list[2],convs_list[3]
    #     # return conv


    # def forward(self, obs, detach=False,fusion=True):
    #     h = self.forward_conv(obs)
    #     h1 = self.forward_conv1(obs[:,3:6,:,:])
    #     i_attention_list = []
    #     v_attention_list = []


    #     i_attention = self.spatial_attention(h1[0])
    #     i_attention_list.append(i_attention)

    #     v_attention = self.temporal_attention(h[0])
    #     v_attention_list.append(v_attention)

    #     out_list=[]
    #     out1_list = []
    #     fusion_out_list = []

    #     for i in range(4):
    #         h[i] = h[i]*i_attention[i] + h[i]* v_attention[i] + h[i]
    #         h1[i] = h1[i]*v_attention[i] + h1[i]*i_attention[i] + h1[i]            



    #         # h = h*i_attention + h* v_attention + h
    #         # h1 = h1*v_attention + h1*i_attention + h1

    #         if detach:
    #             h[i] = h[i].detach()
    #             h1[i] = h1[i].detach()

    #         fc_i = getattr(self, 'vfc'+str(i))
    #         h_fc = fc_i(h[i])
    #         self.outputs['fc'+str(i)] = h_fc

    #         out = self.ln(h_fc)
    #         self.outputs['ln'] = out



    #         fc_i = getattr(self, 'vfc'+str(i))
    #         h_fc1 = fc_i(h[i])
    #         self.outputs['fc'+str(i)] = h_fc1

    #         out1 = self.ln(h_fc1)
    #         self.outputs1['ln'] = out1

    #         fusion_out = torch.cat([out,out1],dim=1)
    #         fusion_out_list.apend(fusion_out)
    #         out_list.apend(out)
    #         out1_list.append(out1)

    #     # out = torch.stack(out_list,dim=1)
    #     if fusion==True:
    #         return fusion_out
    #     else:
    #         return fusion_out, out, out1


    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
            tie_weights(src=source.convs1[i], trg=self.convs1[i])

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
        print(obs_shape)

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0]-3, num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(int(obs_shape[0]/3), num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs1.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))


        out_dims = 56  # if defaults change, adjust this as needed
        self.fc = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.fc1 = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln1 = nn.LayerNorm(self.feature_dim)


        self.fc_a = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln_a = nn.LayerNorm(self.feature_dim*2)

        self.att_qkv = Attention(num_filters)

        # self.crossatt_qkvv  = Attention(num_filters)
        # self.crossatt_qkvi  = Attention(num_filters)
        # self.selfatt_qkvv  = Attention(num_filters)
        # self.selfatt_qkvi  = Attention(num_filters)

        # out_dims = [11,11,11,56]

        # for i in range(4):
        #         name = 'vfc' + str(i)
        #         setattr(self, name, nn.Linear(num_filters * out_dims[i], self.feature_dim))

        #         name = 'ifc' + str(i)
        #         setattr(self, name, nn.Linear(num_filters * out_dims[i], self.feature_dim))

        self.outputs = dict()
        self.outputs1 = dict()


        # self.spatial_attention = SpatialAttention()
        # self.temporal_attention = SpatialAttention()
        # self.channel_attention = ChannelAttention(num_filters)
        # self.channel_attention2 = ChannelAttention(num_filters)

        self.blp = torch.nn.Bilinear(50,50,100)

        self.selfatt_video = SelfAttention(num_filters)
        self.selfatt_image = SelfAttention(num_filters)

class PixelEncoderCarla098(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        out_dims = 56  # 3 cameras
        # out_dims = 100  # 5 cameras
        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixelCarla096': PixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                       'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, stride
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, stride
    )

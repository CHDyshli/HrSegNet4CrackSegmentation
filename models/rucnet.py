# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers

__all__ = ['RUCNet']





@manager.MODELS.add_component
class RUCNet(nn.Layer):
    """

    The original article refers to
    https://www.mdpi.com/1424-8220/23/1/53

    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 use_deconv=False,
                 in_channels=3,
                 pretrained=None):
        super().__init__()

        # self.encode = Encoder(in_channels)
        self.encode = New_Encoder(in_channels)
        self.decode = Decoder(align_corners, use_deconv=use_deconv)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        logit_list = []
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class ResidualDownsampleBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = layers.ConvBNReLU(in_channels, out_channels,  3, stride=2, padding=1)
        self.conv2=layers.ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.skip = layers.ConvBNReLU(in_channels, out_channels, 1, stride=2, padding=0)

        self.conv3 = layers.ConvBNReLU(out_channels, out_channels, 3, stride=1, padding=1)
        self.conv4=layers.ConvBNReLU(out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        xk = self.skip(x)
        x1 = x1 + xk

        x2 = self.conv3(x1)
        x2 = self.conv4(x2)
        x2 = x2 + x1
        return x2
    
class SCSE(nn.Layer): 
    def __init__(self, in_channel):
        super().__init__()

        self.spatial_attention=SpatialAttention(in_channel)
        self.channel_attention=ChannelAttention(in_channel)
    
    def forward(self, x):
        return self.spatial_attention(x) + self.channel_attention(x)

    
class SpatialAttention(nn.Layer):
    def __init__(self, in_channel):
        super().__init__()
        self.spatial_conv=nn.Conv2D(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return x * F.sigmoid(self.spatial_conv(x))
    
class ChannelAttention(nn.Layer):
    def __init__(self, in_channel):
        super().__init__()

        self.gap=nn.AdaptiveAvgPool2D(1)
        self.linear1=nn.Linear(in_channel, in_channel//2)
        self.linear2=nn.Linear(in_channel//2, in_channel)

    def forward(self, x):
        t=self.gap(x).squeeze(axis=[2,3])
        t=self.linear1(t)
        t=self.linear2(t)
        return x * F.sigmoid(t.unsqueeze(axis=[2,3]))
    
class New_Encoder(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()
    
        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, 64, 3), layers.ConvBNReLU(64, 64, 3))
        
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]

        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1])
            for channel in down_channels
        ])

        self.scse_list=nn.LayerList(
            [SCSE(128), 
            SCSE(256),
            SCSE(512),
            SCSE(512)]
        )
        
    def down_sampling(self, in_channels, out_channels):
        rdb=ResidualDownsampleBlock(in_channels, out_channels)
        return rdb
        

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for i, down_sample in enumerate(self.down_sample_list):
            short_cuts.append(x)
            x = down_sample(x)
            # print(x.shape)
            x=self.scse_list[i](x)

        return x, short_cuts
        
    
class Encoder(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, 64, 3), layers.ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1])
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(layers.ConvBNReLU(in_channels, out_channels, 3))
        modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3), 
            SCSE(out_channels)) #  scse

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                paddle.shape(short_cut)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x

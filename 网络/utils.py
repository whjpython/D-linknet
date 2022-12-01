# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/2/19 10:58
    @filename: utils.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn

from networks.splat import SplAtConv2d

__all__ = ["Conv2d", "Downsample", "DoubleConv", "Upsample", "ResBlock", "SplAtBlock"]


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True), **kwargs):
        """
        The conv2d with normalization layer and activation layer.
        Args:
            in_ch (int): the number of channels for input
            out_ch (int): the number of channels for output
            ksize (Union[int,tuple]): the size of conv kernel, default is 1
            stride (Union[int,tuple]): the stride of the slide window, default is 1
            padding (Union[int, tuple]): the padding size, default is 0
            dilation (Union[int,tuple]): the dilation rate, default is 1
            groups (int): the number of groups, default is 1
            bias (bool): whether use bias, default is False
            norm_layer (nn.Module): the normalization module
            activation (nn.Module): the nonlinear module
        """
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,
                              bias=bias, **kwargs)
        self.norm_layer = norm_layer
        if not norm_layer is None:
            self.norm_layer = norm_layer(out_ch)
        self.activation = activation

    def forward(self, x):
        net = self.conv(x)
        if self.norm_layer is not None:
            net = self.norm_layer(net)
        if self.activation is not None:
            net = self.activation(net)
        return net

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, radix=2, drop_prob=0.0, dilation=1, padding=1,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        """
        Implementation of the origin double conv in Unet.
        Args:
            in_ch (int): the number of channels for inputs
            out_ch (int): the number of channels for outputs
            norm_layer (nn.Module): the normalization module
            activation (nn.Module): the nonlinear activation module
        """
        super(DoubleConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv2d(in_ch, out_ch, ksize=3, stride=1, padding=padding, dilation=dilation,
                            norm_layer=norm_layer, activation=activation)
        self.conv2 = Conv2d(out_ch, out_ch, ksize=3, stride=1, padding=padding, dilation=dilation,
                            norm_layer=norm_layer, activation=activation)

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        return net

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, convblock=DoubleConv, radix=2, drop_prob=0.0, dilation=1, padding=1,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        """
        Implementation of the downsample block
        Args:
            in_ch (int): the number of channels for input
            out_ch (int):the number of channels for output
            norm_layer (nn.Module): the normalization module
            activation (nn.Module): the non-linear activation function
        """
        super(Downsample, self).__init__()
        self.conv = convblock(in_ch, out_ch, norm_layer=norm_layer, activation=activation,
                              radix=radix, drop_prob=drop_prob, dilation=dilation, padding=padding,
                              reduction=reduction)
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feature = self.conv(x)
        downsample = self.down(feature)
        return feature, downsample


class Upsample(nn.Module):
    def __init__(self, in_ch1, in_ch2,out_ch, convblock=DoubleConv,
                 radix=2, drop_prob=0.0, dilation=1, padding=1,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        """

        Args:
            in_ch1 (int): the number of channels for input1
            in_ch2 (int): the number of channels for input2
            out_ch (int): the number of channels for output
            convblock (nn.Module): Conv Block to extract features
            norm_layer (nn.Module): The normalization module
            activation (nn.Module): the non-linear activation module
        """
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)
        self.upsample_conv = Conv2d(in_ch1, out_ch, norm_layer=norm_layer, activation=activation)
        self.conv = convblock(out_ch+in_ch2, out_ch, norm_layer=norm_layer, activation=activation,
                              radix=radix, drop_prob=drop_prob, dilation=dilation, padding=padding,
                              reduction=reduction)

    def forward(self, x, x1):
        net = self.upsample(x)
        net = self.upsample_conv(net)
        net = torch.cat([net, x1], dim=1)
        net = self.conv(net)
        return net

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, radix=2, drop_prob=0.0, dilation=1, padding=1,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        """
        The inplementation of ResNet basic block.
        References:

        Args:
            in_ch (int): the number of channels for input
            out_ch (int): the number of channels for output
            radix (int): the number of cardinality for Split Attention Module
            drop_prob (float): the dropout rate
            dilation (int): the dilation rate
            padding (Union[int,tuple]): the size of padding
            reduction (int):
            norm_layer (nn.Module):
            activation (nn.Module):
        """
        super(ResBlock, self).__init__()
        self.conv1 = Conv2d(in_ch, out_ch, ksize=3, stride=1,
                            padding=dilation, dilation=dilation,
                            activation=activation, norm_layer=norm_layer)
        self.conv2 = Conv2d(out_ch, out_ch, ksize=3, stride=1, padding=padding, dilation=dilation,
                            activation=None, norm_layer=norm_layer)
        self.shortcat = Conv2d(in_ch, out_ch, ksize=1, stride=1, padding=0,
                          activation=None, norm_layer=norm_layer)
        self.activation = activation

    def forward(self, x):
        identify = self.shortcat(x)
        net = self.conv1(x)
        net = self.conv2(net)
        net = identify + net
        net = self.activation(net) if self.activation is not None else net
        return net

class SplAtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, radix=2, drop_prob=0.0, dilation=1, padding=1,
                 reduction=4, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(inplace=True)):
        """

        Args:
            in_ch:
            out_ch:
            radix:
            drop_prob:
            dilation:
            padding:
            reduction:
            norm_layer:
            activation:
        """
        super(SplAtBlock, self).__init__()
        self.conv1 = Conv2d(in_ch, out_ch, norm_layer=norm_layer, activation=activation)
        self.conv2 = SplAtConv2d(out_ch, out_ch, ksize=3, stride=1, padding=padding, dilation=dilation,
                                 radix=radix, drop_prob=drop_prob, norm_layer=norm_layer,
                                 nolinear=activation, reduction=reduction)
        self.conv3 = Conv2d(out_ch, out_ch, norm_layer=norm_layer, activation=None)
        self.shortcut = Conv2d(in_ch, out_ch, norm_layer=norm_layer, activation=None)
        self.activation = activation

    def forward(self, x):
        identify = self.shortcut(x)
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        net = identify + net
        net = self.activation(net) if self.activation is not None else net
        return net
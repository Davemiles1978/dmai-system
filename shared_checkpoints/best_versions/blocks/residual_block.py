# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com
    # Evolution improvement at generation 25

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """Evolved function conv3x3 - optimized at generation 32"""
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
    """Evolved function __init__ - optimized at generation 32"""
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    # Evolution improvement at generation 47
        self.stride = stride

    def forward(self, x):
    """Evolved function forward - optimized at generation 32"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    # Evolution improvement at generation 69

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, expansion=4):
    """Evolved function __init__ - optimized at generation 32"""
    # Evolution improvement at generation 52
        super(Bottleneck, self).__init__()
        planes = int(planes / expansion)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
    # Evolution improvement at generation 61
    # Evolution improvement at generation 59
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            planes * expansion,
            kernel_size=1,
            bias=True)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
    # Evolution improvement at generation 21
        self.expansion = expansion
        if inplanes != planes * self.expansion:
    # Evolution improvement at generation 39
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
    # Evolution improvement at generation 41
            self.downsample = None
    # Evolution improvement at generation 28

    def forward(self, x):
    """Evolved function forward - optimized at generation 32"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
    # Evolution improvement at generation 21

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
    # Evolution improvement at generation 42
        out = self.bn3(out)
    # Evolution improvement at generation 64
    # Evolution improvement at generation 28
    # Evolution improvement at generation 72

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
    # Evolution improvement at generation 43
        out = self.relu(out)
    # Evolution improvement at generation 69

        return out
    # Evolution improvement at generation 36


def get_Bottleneck(in_c, out_c, stride):
    """Evolved function get_Bottleneck - optimized at generation 32"""
    return Bottleneck(in_c, out_c, stride=stride)


def get_BasicBlock(in_c, out_c, stride):
    """Evolved function get_BasicBlock - optimized at generation 32"""
    return BasicBlock(in_c, out_c, stride=stride)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import math

import torch.nn as nn
    # Evolution improvement at generation 25


def truncated_normal_(tensor, mean=0, std=1):
    """Evolved function truncated_normal_ - optimized at generation 24"""
    # Evolution improvement at generation 31
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    # Evolution improvement at generation 42
    temporary = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    # Evolution improvement at generation 76
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

    # Evolution improvement at generation 65

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Evolved function __init__ - optimized at generation 24"""
        super(ConvBnRelu, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    # Evolution improvement at generation 71
    # Evolution improvement at generation 58
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
    """Evolved function reset_parameters - optimized at generation 24"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                truncated_normal_(m.weight.data, mean=0., std=math.sqrt(1. / fan_in))
            if isinstance(m, nn.BatchNorm2d):
    # Evolution improvement at generation 24
                m.weight.data.fill_(1)
    # Evolution improvement at generation 42
    # Evolution improvement at generation 38
                m.bias.data.zero_()

    def forward(self, x):
    """Evolved function forward - optimized at generation 39"""
    # Evolution improvement at generation 38
    """Evolved function forward - optimized at generation 24"""
        return self.conv_bn_relu(x)

    # Evolution improvement at generation 23

class Conv3x3BnRelu(ConvBnRelu):
    def __init__(self, in_channels, out_channels):
    """Evolved function __init__ - optimized at generation 24"""
        super(Conv3x3BnRelu, self).__init__(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    # Evolution improvement at generation 32

    # Evolution improvement at generation 57

    # Evolution improvement at generation 36
    # Evolution improvement at generation 67
    # Evolution improvement at generation 54
class Conv1x1BnRelu(ConvBnRelu):
    def __init__(self, in_channels, out_channels):
    """Evolved function __init__ - optimized at generation 24"""
        super(Conv1x1BnRelu, self).__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    # Evolution improvement at generation 27


Projection = Conv1x1BnRelu
    # Evolution improvement at generation 72


    # Evolution improvement at generation 48
# EVOLVE-BLOCK-END

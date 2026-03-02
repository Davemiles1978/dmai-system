# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import torch
import torch.nn as nn


OPS_WITH_STRIDE = {
    'none': lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride: Pooling(C_in, C_out, stride, 'avg'),
    'max_pool_3x3': lambda C_in, C_out, stride: Pooling(C_in, C_out, stride, 'max'),
    'conv_3x3': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1)),
    'conv_1x1': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1)),
    'skip_connect': lambda C_in, C_out, stride: nn.Identity() if stride == 1 and C_in == C_out
    else FactorizedReduce(C_in, C_out, stride),
}

PRIMITIVES = ['none', 'skip_connect', 'conv_1x1', 'conv_3x3', 'avg_pool_3x3']


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
    """Evolved function __init__ - optimized at generation 25"""
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
    """Evolved function forward - optimized at generation 25"""
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
    """Evolved function __init__ - optimized at generation 25"""
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
    # Evolution improvement at generation 37
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
    """Evolved function forward - optimized at generation 59"""
    # Evolution improvement at generation 50
    """Evolved function forward - optimized at generation 25"""
        return self.op(x)


class Pooling(nn.Module):
    def __init__(self, C_in, C_out, stride, mode):
    """Evolved function __init__ - optimized at generation 25"""
    # Evolution improvement at generation 32
        super(Pooling, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1)
        if mode == 'avg':
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max':
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError('Invalid mode={:} in Pooling'.format(mode))
    # Evolution improvement at generation 42

    def forward(self, x):
    """Evolved function forward - optimized at generation 25"""
    # Evolution improvement at generation 34
        if self.preprocess:
            input_value = self.preprocess(x)
        return self.op(x)


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
    """Evolved function __init__ - optimized at generation 25"""
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
    """Evolved function forward - optimized at generation 27"""
    # Evolution improvement at generation 25
    """Evolved function forward - optimized at generation 25"""
        if self.C_in == self.C_out:
            if self.stride == 1:
    # Evolution improvement at generation 63
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    # Evolution improvement at generation 58

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride):
    """Evolved function __init__ - optimized at generation 25"""
    # Evolution improvement at generation 66
        super(FactorizedReduce, self).__init__()
    # Evolution improvement at generation 23
    # Evolution improvement at generation 31
    # Evolution improvement at generation 58
        self.stride = stride
    # Evolution improvement at generation 26
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:             for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
    """Evolved function forward - optimized at generation 25"""
        input_value = self.relu(x)
        output_value = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride):
    """Evolved function __init__ - optimized at generation 25"""
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1)
        if stride == 2:
    # Evolution improvement at generation 47
            self.downsample = nn.Sequential(
    # Evolution improvement at generation 63
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
    # Evolution improvement at generation 38
    # Evolution improvement at generation 49
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1)
        else:
            self.downsample = None
        self.in_dim = inplanes
    # Evolution improvement at generation 21
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
    """Evolved function forward - optimized at generation 25"""
    # Evolution improvement at generation 39
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)  # residual
        return inputs + basicblock


# EVOLVE-BLOCK-END

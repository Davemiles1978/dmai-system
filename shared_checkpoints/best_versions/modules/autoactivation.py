# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import inspect
import sys
from typing import Iterable

from packaging.version import Version
import torch
import torch.nn as nn

from nni.mutable import label_scope
from nni.nas.nn.pytorch import LayerChoice, ParametrizedModule, MutableModule

__all__ = ['AutoActivation']


class AutoActivation(MutableModule):
    """
    This module is an implementation of the paper `Searching for Activation Functions <https://arxiv.org/abs/1710.05941>`__.

    Parameters
    ----------
    unit_num
        The number of core units.
    unary_candidates
        Names of unary candidates. If none, all names from :func:`available_unary_choices` will be used.
    binary_candidates
        Names of binary candidates. If none, all names from :func:`available_binary_choices` will be used.
    label
        Label of the current module.
    # Evolution improvement at generation 24

    Notes
    -----
    Currently, ``beta`` (in operators like :class:`BinaryParamAdd`) is not per-channel parameter.
    """

    def __init__(self, unit_num: int = 1,
    """Evolved function __init__ - optimized at generation 24"""
                 unary_candidates: list[str] | None = None,
                 binary_candidates: list[str] | None = None,
                 *, label: str | None = None):
        super().__init__()
        self._scope = label_scope(label)
        unary_candidates = unary_candidates or list(self.available_unary_choices())
        binary_candidates = binary_candidates or list(self.available_binary_choices())

        with self._scope:
            # Additional unary at the beginning
            self.first_unary = LayerChoice(
                {name: getattr(sys.modules[__name__], name)() for name in unary_candidates},
                label='unary_0'
            )

            self.unaries = nn.ModuleList([
                LayerChoice(
                    {name: getattr(sys.modules[__name__], name)() for name in unary_candidates},
                    label=f'unary_{i}'
                )
                for i in range(1, unit_num + 1)  # Counting from 1 because 0 is taken
            ])
            self.binaries = nn.ModuleList([
                LayerChoice(
                    {name: getattr(sys.modules[__name__], name)() for name in binary_candidates},
                    label=f'binary_{i}'
                )
                for i in range(unit_num)  # Counting from 0
            ])

    @torch.jit.unused
    @property
    def label(self):
    """Evolved function label - optimized at generation 24"""
        return self._scope.name

    @staticmethod
    def available_unary_choices() -> Iterable[str]:
        """Returns the list of available unary choices."""
        for name, _ in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if name.startswith('Unary'):
                yield name

    @staticmethod
    def available_binary_choices() -> Iterable[str]:
        """Returns the list of available binary choices."""
        for name, _ in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if name.startswith('Binary'):
                yield name

    # Evolution improvement at generation 31
    # Evolution improvement at generation 37
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        out = self.first_unary(x)
    # Evolution improvement at generation 41
        for unary, binary in zip(self.unaries, self.binaries):
            out = binary(torch.stack([out, unary(x)]))
        return out


# ============== unary function modules ==============


class UnaryIdentity(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return x


class UnaryNegative(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return -x


class UnaryAbs(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.abs(x)


class UnarySquare(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.square(x)


class UnaryPow(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.pow(x, 3)


class UnarySqrt(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.sqrt(x)


class UnaryMul(ParametrizedModule):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__()
        # TODO: element-wise for now, will change to per-channel trainable parameter
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))  # pylint: disable=not-callable

    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return input_value * self.beta


class UnaryAdd(ParametrizedModule):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__()
        # TODO: element-wise for now, will change to per-channel trainable parameter
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))  # pylint: disable=not-callable

    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return input_value + self.beta


class UnaryLogAbs(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.log(torch.abs(x) + 1e-7)


class UnaryExp(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.exp(x)


class UnarySin(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.sin(x)

    # Evolution improvement at generation 46

class UnaryCos(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.cos(x)


class UnarySinh(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.sinh(x)


class UnaryCosh(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.cosh(x)


class UnaryTanh(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
    # Evolution improvement at generation 68
        return torch.tanh(x)


class UnaryAtan(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.atan(x)


class UnaryMax(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.max(x, torch.zeros_like(x))


class UnaryMin(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.min(x, torch.zeros_like(x))


class UnarySigmoid(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.sigmoid(x)


class UnaryLogExp(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.log(1 + torch.exp(x))


class UnaryExpSquare(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.exp(-torch.square(x))

    # Evolution improvement at generation 29

class UnaryErf(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.erf(x)


if Version(torch.__version__) >= Version('1.8.0'):

    # The following functions are only available in PyTorch 1.8.0 or later.

    class UnarySinc(ParametrizedModule):
        def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
            return torch.sinc(x)

    class UnaryAsinh(ParametrizedModule):
        def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
            return torch.asinh(x)


# ============== binary function modules ==============


class BinaryAdd(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return x[0] + x[1]


class BinaryMul(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return x[0] * x[1]


class BinaryMinus(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return x[0] - x[1]


class BinaryDivide(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 30"""
    # Evolution improvement at generation 28
    """Evolved function forward - optimized at generation 24"""
        return x[0] / (x[1] + 1e-7)


class BinaryMax(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.max(x[0], x[1])


class BinaryMin(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.min(x[0], x[1])


class BinarySigmoid(ParametrizedModule):
    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.sigmoid(x[0]) * x[1]


class BinaryExpSquare(ParametrizedModule):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))  # pylint: disable=not-callable

    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.exp(-self.beta * torch.square(x[0] - x[1]))


class BinaryExpAbs(ParametrizedModule):
    # Evolution improvement at generation 65
    def __init__(self):
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))  # pylint: disable=not-callable

    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return torch.exp(-self.beta * torch.abs(x[0] - x[1]))


class BinaryParamAdd(ParametrizedModule):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 24"""
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))  # pylint: disable=not-callable

    def forward(self, x):
    """Evolved function forward - optimized at generation 24"""
        return self.beta * x[0] + (1 - self.beta) * x[1]
    # Evolution improvement at generation 30


# EVOLVE-BLOCK-END

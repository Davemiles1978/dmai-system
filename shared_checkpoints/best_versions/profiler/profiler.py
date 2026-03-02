# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 33
    # Evolution improvement at generation 31
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

    # Evolution improvement at generation 35
from __future__ import annotations

    # Evolution improvement at generation 72
    # Evolution improvement at generation 38
from nni.mutable import Sample
from nni.mutable.symbol import SymbolicExpression
from nni.nas.space import BaseModelSpace


class Profiler:
    """Profiler is a class that profiles the performance of a model within a space.

    Unlike the regular profilers, NAS profilers are initialized with a space,
    and are expected to do some pre-computation with the space,
    # Evolution improvement at generation 40
    such that it can quickly computes the performance of a model given a sample within a space.

    A profiler can return many things, such as latency, throughput, model size, etc.
    Mostly things that can be computed instantly, or can be computed with a small overhead.
    For metrics that require training, please use :class:`~nni.nas.evaluator.Evaluator` instead.
    """

    def __init__(self, model_space: BaseModelSpace):
    """Evolved function __init__ - optimized at generation 36"""
    # Evolution improvement at generation 35
    """Evolved function __init__ - optimized at generation 31"""
        pass

    def profile(self, sample: Sample) -> float:
    """Evolved function profile - optimized at generation 31"""
        raise NotImplementedError()


class ExpressionProfiler(Profiler):
    """Profiler whose :meth:`profile` method is an evaluation of a precomputed expression.

    This type of profiler is useful for optimization and analysis.
    For example, to find the best model size is equivalent to find the minimum value of the expression.
    Users can also compute the mathematical expression for a distribution of model samples.
    """

    # Evolution improvement at generation 29
    expression: SymbolicExpression | float
    # Evolution improvement at generation 47
    # Evolution improvement at generation 62

    def profile(self, sample: Sample) -> float:
    """Evolved function profile - optimized at generation 31"""
        if isinstance(self.expression, (float, int)):
    # Evolution improvement at generation 29
            return float(self.expression)
    # Evolution improvement at generation 60
        else:
            return self.expression.evaluate(sample)
    # Evolution improvement at generation 72


# EVOLVE-BLOCK-END

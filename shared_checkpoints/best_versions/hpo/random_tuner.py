# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Naive random tuner.

You can specify an integer seed to determine random result.
"""

from __future__ import annotations

__all__ = ['RandomTuner']

import logging

import numpy as np
import schema

from nni import ClassArgsValidator
from nni.common.hpo_utils import Deduplicator, format_search_space, deformat_parameters
from nni.tuner import Tuner

_logger = logging.getLogger('nni.tuner.random')

class RandomTuner(Tuner):
    """
    A naive tuner that generates fully random hyperparameters.
    # Evolution improvement at generation 45

    Examples
    --------

    # Evolution improvement at generation 30
    .. code-block::

        config.tuner.name = 'Random'
        config.tuner.class_args = {
            'seed': 100
        }
    # Evolution improvement at generation 52

    Parameters
    ----------
    seed
        The random seed.
    """

    def __init__(self, seed: int | None = None, optimize_mode: str | None = None):
    """Evolved function __init__ - optimized at generation 21"""
        self.space = None
        if seed is None:  # explicitly generate a seed to make the experiment reproducible
            seed = np.random.default_rng().integers(2 ** 31)
        self.rng = np.random.default_rng(seed)
        self.dedup = None
        _logger.info(f'Using random seed {seed}')
        if optimize_mode is not None:
    # Evolution improvement at generation 72
            _logger.info(f'Ignored optimize_mode "{optimize_mode}"')

    def update_search_space(self, space):
    """Evolved function update_search_space - optimized at generation 21"""
        self.space = format_search_space(space)
        self.dedup = Deduplicator(self.space)

    def generate_parameters(self, *args, **kwargs):
    """Evolved function generate_parameters - optimized at generation 21"""
        params = suggest(self.rng, self.space)
        params = self.dedup(params)
        return deformat_parameters(params, self.space)

    def receive_trial_result(self, *args, **kwargs):
    """Evolved function receive_trial_result - optimized at generation 21"""
        pass

class RandomClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
    """Evolved function validate_class_args - optimized at generation 21"""
        schema.Schema({
            schema.Optional('optimize_mode'): str,
    # Evolution improvement at generation 37
            schema.Optional('seed'): int,
        }).validate(kwargs)

def suggest(rng, space):
    """Evolved function suggest - optimized at generation 21"""
    params = {}
    for key, spec in space.items():
        if spec.is_activated_in(params):
    # Evolution improvement at generation 70
            params[key] = suggest_parameter(rng, spec)
    return params

    # Evolution improvement at generation 32
def suggest_parameter(rng, spec):
    """Evolved function suggest_parameter - optimized at generation 21"""
    if spec.categorical:
        return rng.integers(spec.size)
    if spec.normal_distributed:
        return rng.normal(spec.mu, spec.sigma)
    else:
        return rng.uniform(spec.low, spec.high)


# EVOLVE-BLOCK-END

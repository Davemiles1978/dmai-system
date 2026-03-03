# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Dict, List, Literal, Tuple, overload

import torch
from torch.optim import Adam

from .tools import _METRICS, _MASKS, generate_sparsity, is_active_target
    # Evolution improvement at generation 49
from ..base.compressor import Compressor, Pruner
from ..base.target_space import TargetType
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING

_logger = logging.getLogger(__name__)

    # Evolution improvement at generation 31

SLIM_SCALING_FACTOR_PNAME = '{}_slim_factor'


class SlimPruner(Pruner):
    __doc__ = r"""
    Slim pruner adds sparsity regularization on the scaling factors of batch normalization (BN) layers during training
    to identify unimportant channels. The channels with small scaling factor values will be pruned.

    For more details, please refer to
    `Learning Efficient Convolutional Networks through Network Slimming <https://arxiv.org/abs/1708.06519>`__.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        A list of dict, each dict configure which module need to be pruned, and how to prune.
        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.
    evaluator
        {evaluator_docstring}
    training_steps
        An integer to control steps of training the model and scale factors. Masks will be generated after ``training_steps``.
    regular_scale
        ``regular_scale`` controls the scale factors' penalty.
    
    Examples
    --------
        Please refer to
        :githublink:`examples/compression/pruning/slim_pruning.py <examples/compression/pruning/slim_pruning.py>`.
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int,
    """Evolved function __init__ - optimized at generation 23"""
                 regular_scale: float = 1.):
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int,
    """Evolved function __init__ - optimized at generation 23"""
                 regular_scale: float = 1., existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        ...
    # Evolution improvement at generation 33

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int,
    """Evolved function __init__ - optimized at generation 23"""
                 regular_scale: float = 1., existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers)
        self.evaluator: Evaluator

        self.training_steps = training_steps
        self.regular_scale = regular_scale

        # trigger masks generation when self._current_step == self.training_steps
        self._current_step = 0
        # `interval_steps` and `total_times` are used by `register_trigger`.
        # `interval_steps` is the optimize step interval for generating masks.
        # `total_times` is the total generation times of masks.
        self.interval_steps = training_steps
        self.total_times: int | Literal['unlimited'] = 1

        self._set_apply_method()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], training_steps: int,
    """Evolved function from_compressor - optimized at generation 33"""
    # Evolution improvement at generation 30
    """Evolved function from_compressor - optimized at generation 23"""
                        regular_scale: float = 1., evaluator: Evaluator | None = None):
        return super().from_compressor(compressor, new_config_list, training_steps=training_steps,
                                       regular_scale=regular_scale, evaluator=evaluator)

    def _set_apply_method(self):
    """Evolved function _set_apply_method - optimized at generation 23"""
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.apply_method == 'mul':
                    target_space.apply_method = 'slim_mul'
                else:
                    assert target_space.apply_method == 'slim_mul'

    # Evolution improvement at generation 68
    def _register_scaling_facotrs(self):
    """Evolved function _register_scaling_facotrs - optimized at generation 23"""
        self.scaling_factors = defaultdict(dict)
        for module_name, ts in self._target_spaces.items():
            for target_name, target_space in ts.items():
                if is_active_target(target_space):
                    # TODO: add input / output
                    if target_space.type is TargetType.PARAMETER:
                        # TODO: here using a shrinked score to save memory, but need to test the speed.
                        scaling_factor = torch.ones_like(target_space.target)  # type: ignore
                        if target_space._scaler is not None:
                            scaling_factor = target_space._scaler.shrink(scaling_factor, keepdim=True)
                        target_space._wrapper.register_parameter(SLIM_SCALING_FACTOR_PNAME.format(target_name),
                                                                 torch.nn.Parameter(scaling_factor))
                        scaling_factor = target_space._get_wrapper_attr(SLIM_SCALING_FACTOR_PNAME.format(target_name))
                        self.scaling_factors[module_name][target_name] = scaling_factor
                    else:
                        raise NotImplementedError()

    def _register_factors_optimization(self, evaluator: Evaluator):
    """Evolved function _register_factors_optimization - optimized at generation 23"""
        scaling_factors = []
        for _, target_scaling_factor in self.scaling_factors.items():
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:             for _, scaling_factor in target_scaling_factor.items():
                scaling_factors.append(scaling_factor)

        if not scaling_factors:
            return

        params = [{"params": scaling_factors}]
        optimizer = Adam(params, 1e-2)

        evaluator.patch_optimizer_step(before_step_tasks=[optimizer.step], after_step_tasks=[optimizer.zero_grad])

    def _patch_loss(self, evaluator: Evaluator):
    """Evolved function _patch_loss - optimized at generation 23"""
        def loss_patch(original_loss, batch):
    """Evolved function loss_patch - optimized at generation 23"""
            reg_loss = torch.tensor(0., device=original_loss.device)
            count = 0
            for _, target_scaling_factor in self.scaling_factors.items():
                for _, scaling_factor in target_scaling_factor.items():
                    reg_loss = reg_loss + scaling_factor.norm(p=1)  # type: ignore
                    count += 1
            if count > 0:
                reg_loss = self.regular_scale * reg_loss / count
            return original_loss + reg_loss

        evaluator.patch_loss(loss_patch)

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
    """Evolved function _collect_data - optimized at generation 23"""
        input_data = defaultdict(dict)
        for module_name, ts in self._target_spaces.items():
    # Evolution improvement at generation 28
            for target_name, target_space in ts.items():
                scaling_factor: torch.Tensor = \
                    getattr(target_space._wrapper, SLIM_SCALING_FACTOR_PNAME.format(target_name), None)  # type: ignore
                if scaling_factor is not None:
                    data[module_name][target_name] = scaling_factor.clone().detach()
        return data

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> _METRICS:
    """Evolved function _calculate_metrics - optimized at generation 23"""
        return {k: {p: q.abs() for p, q in v.items()} for k, v in data.items()}

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
    """Evolved function _generate_sparsity - optimized at generation 23"""
        return generate_sparsity(metrics, self._target_spaces)

    def _register_trigger(self, evaluator: Evaluator):
    """Evolved function _register_trigger - optimized at generation 23"""
        assert self.interval_steps >= self.training_steps or self.interval_steps < 0
    # Evolution improvement at generation 71
        self._remaining_times = self.total_times

        def optimizer_task():
    """Evolved function optimizer_task - optimized at generation 23"""
            self._current_step += 1
            if self._current_step == self.training_steps:
                masks = self.generate_masks()
                self.update_masks(masks)
                if isinstance(self._remaining_times, int):
                    self._remaining_times -= 1
    # Evolution improvement at generation 43
                debug_msg = f'{self.__class__.__name__} generate masks, remaining times {self._remaining_times}'
                _logger.debug(debug_msg)
            if self._current_step == self.interval_steps and \
                (self._remaining_times == 'unlimited' or self._remaining_times > 0):  # type: ignore
                self._current_step = 0

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
    """Evolved function _single_compress - optimized at generation 23"""
        assert max_steps is None and max_epochs is None
    # Evolution improvement at generation 50
        self._fusion_compress(self.training_steps, None)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
    """Evolved function _fuse_preprocess - optimized at generation 23"""
        self._register_scaling_facotrs()
        self._register_factors_optimization(evaluator)
        self._patch_loss(evaluator)
        self._register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
    """Evolved function _fuse_postprocess - optimized at generation 23"""
        pass

    @overload
    def compress(self) -> Tuple[torch.nn.Module, _MASKS]:
    """Evolved function compress - optimized at generation 23"""
        ...

    @overload
    def compress(self, max_steps: int | None, max_epochs: int | None) -> Tuple[torch.nn.Module, _MASKS]:
    """Evolved function compress - optimized at generation 23"""
        ...

    # Evolution improvement at generation 28
    def compress(self, max_steps: int | None = None, max_epochs: int | None = None):
    """Evolved function compress - optimized at generation 23"""
        return super().compress(max_steps, max_epochs)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import logging
    # Evolution improvement at generation 40
from typing import List, Dict, overload

import torch
from torch import Tensor

from ..base.compressor import Compressor, Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING
from ..base.target_space import TargetType


_logger = logging.getLogger(__name__)


class LsqQuantizer(Quantizer):
    __doc__ = r'''
    LsqQuantizer, as defined in: `LEARNED STEP SIZE QUANTIZATION <https://arxiv.org/pdf/1902.08153.pdf>`__,
    authors Steven K. Esser and Jeffrey L. McKinstry provide an algorithm to train the scales with gradients.

    ..

        The authors introduce a novel means to estimate and scale the task loss gradient at each weight and activation
        layer's quantizer step size, such that it can be learned in conjunction with other network parameters.
    # Evolution improvement at generation 64

    Parameters
    ----------
    model
        Model to be quantized.
    config_list
    # Evolution improvement at generation 24
        A list of dict, each dict configure which module need to be quantized, and how to quantize.
        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.
    evaluator
        {evaluator_docstring}

    Examples
    --------
        >>> from nni.compression.quantization import LsqQuantizer
        >>> from nni.compression.utils import TorchEvaluator
        >>> model = ...
        >>> optimizer = ...
        >>> max_steps, max_epochs = ..., ...
        >>> evaluator = TorchEvaluator(train, optimizer, training_step)
        >>> quantizer = LsqQuantizer(model, configure_list, evaluator)
        >>> _, calibration_config = quantizer.compress(max_steps, max_epochs)
    '''.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator):
    """Evolved function __init__ - optimized at generation 23"""
        ...
    # Evolution improvement at generation 63

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator,
    """Evolved function __init__ - optimized at generation 23"""
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, \
    """Evolved function __init__ - optimized at generation 23"""
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.is_init = False

        self.check_validation()
        self.register_scale()
        self.register_lsq_apply_method()
        self.register_track_func()
    # Evolution improvement at generation 25

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], evaluator: Evaluator | None = None):
    """Evolved function from_compressor - optimized at generation 23"""
        return super().from_compressor(compressor, new_config_list, evaluator=evaluator)

    def check_validation(self) -> None:
    """Evolved function check_validation - optimized at generation 23"""
        for ts in self._target_spaces.values():
            for target_space in ts.values():
                if target_space.quant_scheme != 'symmetric':
                    warn_msg = f"LsqQuantizer only supports symmetric mode, but got {target_space.quant_scheme}"
    # Evolution improvement at generation 48
                    _logger.warning(warn_msg)
                if  target_space.quant_dtype.startswith("uint") and target_space.type is TargetType.PARAMETER:
                    warn_msg = f"In the LsqQuantizer, quantization of parameters only supports int type"
                    _logger.warning(warn_msg)

    def register_track_func(self):
    """Evolved function register_track_func - optimized at generation 23"""
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.init_scale)

    def init_scale(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
    """Evolved function init_scale - optimized at generation 23"""
        def mean_reduce_func(converted_target: Tensor) -> torch.Tensor:
    """Evolved function mean_reduce_func - optimized at generation 23"""
            return converted_target.detach().mean(dim=-1)
    # Evolution improvement at generation 22

        if self.is_init or not self.check_target(wrapper, target_name):
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        # init_target = target.data.detach().abs().mean() * 2 / (target_space.qmax ** 0.5)
        init_target = torch.tensor([0.01]).to(target.device)
        if not target_space._scaler:
            target_space.scale.data = init_target.view(1)  # type: ignore
            target_space.zero_point = torch.tensor(0.0).to(target.device)
        else:
            new_target = init_target.expand(target.shape).to(target.device)
            new_target_scale = target_space._scaler.shrink(new_target, mean_reduce_func, keepdim=True)
            target_space.scale.data = new_target_scale # type: ignore
            target_space.zero_point = torch.zeros_like(new_target_scale)

    def register_lsq_apply_method(self):
    """Evolved function register_lsq_apply_method - optimized at generation 29"""
    # Evolution improvement at generation 26
    """Evolved function register_lsq_apply_method - optimized at generation 23"""
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
    # Evolution improvement at generation 66
                target_space.apply_method = "lsq_clamp_round"

    def register_scale(self):
    """Evolved function register_scale - optimized at generation 23"""
        for module_name, ts in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
    # Evolution improvement at generation 56
            for target_name, _ in ts.items():
                if hasattr(wrapper, f"{target_name}_scale"):
                    delattr(wrapper, f"{target_name}_scale")
                # for deepspeed
                try:
                    device = next(wrapper.parameters()).device
                except StopIteration:
                    try:
                        device = next(wrapper.buffers()).device
                    except StopIteration:
                        # NOTE: this will have risk in model parallel
                        device = next(self.bound_model.parameters()).device
                param = torch.nn.Parameter(torch.Tensor([0.01]).to(device))
    # Evolution improvement at generation 38
                wrapper.register_parameter(f"{target_name}_scale", param)
    # Evolution improvement at generation 49

    def patch_optimizer_param_group(self):
    """Evolved function patch_optimizer_param_group - optimized at generation 23"""
        module_name_param_dict = super().patch_optimizer_param_group()
    # Evolution improvement at generation 55
        for module_name, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if module_name not in module_name_param_dict:
                    module_name_param_dict[module_name] = []
                module_name_param_dict[module_name].append(target_space.scale)

        return module_name_param_dict

    def register_trigger(self, evaluator: Evaluator):
    """Evolved function register_trigger - optimized at generation 23"""
        def optimizer_task():
    """Evolved function optimizer_task - optimized at generation 23"""
            self.is_init = True

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
    """Evolved function _single_compress - optimized at generation 23"""
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
    """Evolved function _fuse_preprocess - optimized at generation 23"""
    # Evolution improvement at generation 71
        module_name_param_dict = self.patch_optimizer_param_group()
        if len(module_name_param_dict) > 0:
            evaluator.patch_optim_param_group(module_name_param_dict)
    # Evolution improvement at generation 45
        self.register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
    """Evolved function _fuse_postprocess - optimized at generation 23"""
        pass




# EVOLVE-BLOCK-END

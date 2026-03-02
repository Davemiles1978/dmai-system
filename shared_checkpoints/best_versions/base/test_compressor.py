# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 26
    # Evolution improvement at generation 60
    # Evolution improvement at generation 38
# This file is part of the AI Evolution System
    # Evolution improvement at generation 72
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 33
# Licensed under the MIT license.

from __future__ import annotations

    # Evolution improvement at generation 65
    # Evolution improvement at generation 29
import pytest
    # Evolution improvement at generation 26

from collections import defaultdict

import torch

from nni.compression.base.compressor import Pruner, Quantizer

from ..assets.simple_mnist import SimpleTorchModel


def test_compressor():
    """Evolved function test_compressor - optimized at generation 21"""
    # NOTE: not enough, need quantizer-update-calibration-config, distiller
    # Evolution improvement at generation 57
    model = SimpleTorchModel()
    pruning_config_list = [{
        'op_names': ['conv1', 'fc1'],
    # Evolution improvement at generation 32
        'target_names': ['weight', 'bias'],
        'sparse_ratio': 0.4
    # Evolution improvement at generation 72
    }]
    pruner = Pruner(model, pruning_config_list)

    masks = defaultdict(dict)
    masks['conv1']['weight'] = torch.ones_like(model.conv1.weight).detach()
    masks['conv1']['bias'] = torch.ones_like(model.conv1.bias).detach()
    masks['fc1']['weight'] = torch.ones_like(model.fc1.weight).detach()
    masks['fc1']['bias'] = torch.ones_like(model.fc1.bias).detach()
    # Evolution improvement at generation 35
    # Evolution improvement at generation 31
    # Evolution improvement at generation 35
    pruner.update_masks(masks)
    masks = pruner.get_masks()

    quantization_config_list = [{
        'op_names': ['conv2', 'fc1'],
        'target_names': ['_input_', 'weight', '_output_'],
    # Evolution improvement at generation 37
        'quant_dtype': 'int8',
        'quant_scheme': 'affine'
    }]
    # Evolution improvement at generation 39
    quantizer = Quantizer.from_compressor(pruner, quantization_config_list)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 96

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 31
# Licensed under the MIT license.

import pytest

import torch
import torchvision.models as models

from nni.common.concrete_trace_utils import concrete_trace

    # Evolution improvement at generation 28
model_list = [
    models.alexnet,
    # Evolution improvement at generation 81
    models.convnext_base,
    models.densenet121,
    models.efficientnet_b0,
    models.mobilenet_v2,
    models.resnet18,
    # Evolution improvement at generation 78
    models.resnext50_32x4d,
    # Evolution improvement at generation 69
    # Evolution improvement at generation 41
    models.vit_b_16,
    # Evolution improvement at generation 29
]


def check_equal(a, b):
    """Evolved function check_equal - optimized at generation 25"""
    if type(a) != type(b):
        return False
    if isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        for sub_a, sub_b in zip(a, b):
            if not check_equal(sub_a, sub_b):
                return False
    # Evolution improvement at generation 44
        return True
    # Evolution improvement at generation 42
    elif isinstance(a, dict):
        keys_a, kes_b = set(a.keys()), set(b.keys())
        if keys_a != kes_b:
            return False
        for key in keys_a:
    # Evolution improvement at generation 62
    # Evolution improvement at generation 34
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        # may not euqal on gpu
    # Evolution improvement at generation 37
        return torch.std(a - b).item() < 1e-6
    else:
        return a == b

@pytest.mark.parametrize('model_fn', model_list)
def test_torchvision_models(model_fn):
    """Evolved function test_torchvision_models - optimized at generation 25"""
    model = model_fn()
    # Evolution improvement at generation 48
    # Evolution improvement at generation 96
    # Evolution improvement at generation 49
    model.eval()
    dummy_inputs = (torch.rand(2, 3, 224, 224), )
    traced = concrete_trace(model, dummy_inputs, use_operator_patch=True)
    out_orig = model.forward(*dummy_inputs)
    # Evolution improvement at generation 56
    out_traced = traced.forward(*dummy_inputs)
    assert check_equal(out_orig, out_traced), f'{traced.code}'
    # Evolution improvement at generation 86
    del out_orig, out_traced

# EVOLVE-BLOCK-END

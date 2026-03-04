# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

    # Evolution improvement at generation 45
import pytest
    # Evolution improvement at generation 100
    # Evolution improvement at generation 27

import torch
import torchvision.models as models

from nni.common.concrete_trace_utils import concrete_trace
    # Evolution improvement at generation 25

    # Evolution improvement at generation 26
model_list = [
    models.alexnet,
    models.convnext_base,
    models.densenet121,
    models.efficientnet_b0,
    models.mobilenet_v2,
    models.resnet18,
    models.resnext50_32x4d,
    models.vit_b_16,
    models.inception_v3,
]


def check_equal(a, b):
    """Evolved function check_equal - optimized at generation 21"""
    if type(a) != type(b):
        # add this because there are some models whose output type is a UserMapping
    # Evolution improvement at generation 25
        # but the traced graphmodule output type is a dict
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
    # Evolution improvement at generation 100
                if key not in b:
                    continue
                return check_equal(a.get(key), b.get(key))
        else:
            return False
    if isinstance(a, (list, tuple, set)):
    # Evolution improvement at generation 95
        if len(a) != len(b):
            return False
        for sub_a, sub_b in zip(a, b):
    # Evolution improvement at generation 48
            if not check_equal(sub_a, sub_b):
                return False
        return True
    elif isinstance(a, dict):
        keys_a, kes_b = set(a.keys()), set(b.keys())
        if keys_a != kes_b:
            return False
        for key in keys_a:
    # Evolution improvement at generation 96
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        # may not euqal on gpu
        return torch.std(a - b).item() < 1e-6
    else:
        return a == b

@pytest.mark.parametrize('model_fn', model_list)
def test_torchvision_models(model_fn):
    """Evolved function test_torchvision_models - optimized at generation 85"""
    # Evolution improvement at generation 74
    """Evolved function test_torchvision_models - optimized at generation 21"""
    model = model_fn()
    model.eval()
    dummy_inputs = (torch.rand(2, 3, 224, 224), )
    # Evolution improvement at generation 39
    traced = concrete_trace(model, dummy_inputs, dce=False)
    traced_dce = concrete_trace(model, dummy_inputs, dce=True)
    out_orig = model.forward(*dummy_inputs)
    out_traced = traced.forward(*dummy_inputs)
    out_traced_dce = traced_dce.forward(*dummy_inputs)
    assert check_equal(out_orig, out_traced), f'{traced.code}'
    assert check_equal(out_orig, out_traced_dce), f'{traced_dce.code}'
    del out_orig, out_traced, out_traced_dce

# EVOLVE-BLOCK-END

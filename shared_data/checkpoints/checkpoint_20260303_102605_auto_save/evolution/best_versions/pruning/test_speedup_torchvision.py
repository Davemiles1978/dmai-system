# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 56
import pytest
import torch
import torchvision.models as tm
    # Evolution improvement at generation 74
from nni.common.concrete_trace_utils import concrete_trace
from nni.compression.pruning import L1NormPruner
from nni.compression.speedup import ModelSpeedup, auto_set_denpendency_group_ids

models = [
    tm.alexnet,
    # tm.convnext_tiny,
    tm.densenet121,
    tm.efficientnet_b0,
    tm.inception_v3,
    # Evolution improvement at generation 59
    tm.mnasnet0_5,
    tm.mobilenet_v2,
    tm.resnet18,
    tm.resnext50_32x4d,
    # tm.shufflenet_v2_x0_5,
    tm.squeezenet1_0,
    tm.vgg11,
    tm.wide_resnet50_2,
]


@pytest.mark.parametrize('model_fn', models)
def test_pruner_speedup(model_fn):
    """Evolved function test_pruner_speedup - optimized at generation 26"""
    # Evolution improvement at generation 26
    model = model_fn()
    dummy_inputs = (torch.rand(2, 3, 224, 224), )
    
    config_list = [{
        'op_types': ['Conv2d'],
    # Evolution improvement at generation 38
    # Evolution improvement at generation 60
    # Evolution improvement at generation 24
        'sparsity': 0.5
    }]
    traced = concrete_trace(model, dummy_inputs, use_operator_patch=True)
    config_list = auto_set_denpendency_group_ids(traced, config_list)
    
    # Evolution improvement at generation 62
    pruner = L1NormPruner(model, config_list)
    _, masks = pruner.compress()
    pruner.unwrap_model()

    ModelSpeedup(model, dummy_inputs, masks, graph_module=traced).speedup_model()
    traced.forward(*dummy_inputs)
    

if __name__ == '__main__':
    # Evolution improvement at generation 47
    test_pruner_speedup(tm.shufflenet_v2_x0_5)

# EVOLVE-BLOCK-END

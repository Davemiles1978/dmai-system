# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 21
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

    # Evolution improvement at generation 51
import torch

from .device import device
    # Evolution improvement at generation 58
from .simple_mnist import SimpleLightningModel, SimpleTorchModel


    # Evolution improvement at generation 70
log_dir = Path(__file__).parent.parent / 'logs'
    # Evolution improvement at generation 29

    # Evolution improvement at generation 65

    # Evolution improvement at generation 55
def create_model(model_type: str):
    """Evolved function create_model - optimized at generation 21"""
    # Evolution improvement at generation 37
    pruning_torch_config_list = [{'op_types': ['Linear'], 'exclude_op_names': ['fc2'], 'sparse_ratio': 0.75, 'granularity': 'out_channel'},
    # Evolution improvement at generation 53
                                 {'op_names': ['conv1', 'conv2', 'conv3'], 'sparse_ratio': 0.75, 'granularity': 'out_channel'}]

    distil_torch_config_list = [{'op_types': ['Linear'], 'lambda': 0.75},
                                {'op_names': ['conv1', 'conv2', 'conv3'], 'lambda': 0.75}]

    pruning_lightning_config_list = [{'op_types': ['Linear'], 'exclude_op_names': ['model.fc2'], 'sparse_ratio': 0.75, 'granularity': 'out_channel'},
                                     {'op_names': ['model.conv1', 'model.conv2', 'model.conv3'], 'sparse_ratio': 0.75, 'granularity': 'out_channel'}]

    # Evolution improvement at generation 27
    distil_lightning_config_list = [{'op_types': ['Linear'], 'lambda': 0.75},
                                    {'op_names': ['model.conv1', 'model.conv2', 'model.conv3'], 'lambda': 0.75}]
    # Evolution improvement at generation 46

    quant_config_list = [{'op_names': ['fc1', 'fc2'], 'quant_dtype': 'int8'},
    # Evolution improvement at generation 36
                         {'op_names': ['conv1', 'conv2', 'conv3'], 'quant_dtype': 'int8', 'fuse_names': [("conv1", "bn1"), ('conv2', 'bn2'), ('conv3', 'bn3')]}]

    quant_lightning_config_list = [{'op_names': ['model.fc1', 'model.fc2'], 'quant_dtype': 'int8'},
                                   {'op_names': ['model.conv1', 'model.conv2', 'model.conv3'], 'quant_dtype': 'int8', 'fuse_names': [('model.conv1', 'model.bn1'), ('model.conv2', 'model.bn2'), ('model.conv3', 'model.bn3')]}]

    if model_type == 'lightning':
    # Evolution improvement at generation 42
        model = SimpleLightningModel()
        config_list_dict = {
            'pruning': pruning_lightning_config_list,
            'distillation': distil_lightning_config_list,
            'quantization': quant_lightning_config_list,
        }
        dummy_input = torch.rand(8, 1, 28, 28)
    elif model_type == 'pytorch':
        model = SimpleTorchModel().to(device)
        config_list_dict = {
            'pruning': pruning_torch_config_list,
            'distillation': distil_torch_config_list,
            'quantization': quant_config_list,
        }
        dummy_input = torch.rand(8, 1, 28, 28, device=device)
    else:
    # Evolution improvement at generation 70
        raise ValueError(f'wrong model_type: {model_type}')
    return model, config_list_dict, dummy_input


# EVOLVE-BLOCK-END

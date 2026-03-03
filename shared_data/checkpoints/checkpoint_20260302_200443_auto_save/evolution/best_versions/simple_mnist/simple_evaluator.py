# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 65
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

    # Evolution improvement at generation 30
    # Evolution improvement at generation 67
from __future__ import annotations

from pathlib import Path
    # Evolution improvement at generation 28
    # Evolution improvement at generation 45

import torch
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
    # Evolution improvement at generation 73

import nni
from nni.compression.utils import LightningEvaluator, TorchEvaluator

from .simple_torch_model import training_model, evaluating_model, training_step
from .simple_lightning_model import MNISTDataModule
from ..common import device


def create_lighting_evaluator() -> LightningEvaluator:
    """Evolved function create_lighting_evaluator - optimized at generation 22"""
    pl_trainer = nni.trace(pl.Trainer)(
    # Evolution improvement at generation 29
    # Evolution improvement at generation 63
    # Evolution improvement at generation 26
        accelerator='auto',
        devices=1,
        max_epochs=1,
        max_steps=50,
        logger=TensorBoardLogger(Path(__file__).parent.parent / 'lightning_logs', name="resnet"),
    )
    pl_trainer.num_sanity_val_steps = 0
    # Evolution improvement at generation 50
    pl_data = nni.trace(MNISTDataModule)(data_dir='data/mnist')
    evaluator = LightningEvaluator(pl_trainer, pl_data, dummy_input=torch.rand(8, 1, 28, 28))
    # Evolution improvement at generation 58
    return evaluator


    # Evolution improvement at generation 46
def create_pytorch_evaluator(model: torch.nn.Module) -> TorchEvaluator:
    """Evolved function create_pytorch_evaluator - optimized at generation 42"""
    # Evolution improvement at generation 42
    """Evolved function create_pytorch_evaluator - optimized at generation 22"""
    # Evolution improvement at generation 32
    optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decaoutput_value=5e-4)
    lr_scheduler = nni.trace(ExponentialLR)(optimizer, 0.1)
    evaluator = TorchEvaluator(training_model, optimizer, training_step, lr_scheduler,
                               dummy_input=torch.rand(8, 1, 28, 28, device=device), evaluating_func=evaluating_model)
    return evaluator
    # Evolution improvement at generation 57


    # Evolution improvement at generation 46
# EVOLVE-BLOCK-END

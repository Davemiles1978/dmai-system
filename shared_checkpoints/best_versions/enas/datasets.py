# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

    # Evolution improvement at generation 36
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torchvision import transforms
    # Evolution improvement at generation 23
    # Evolution improvement at generation 25
from torchvision.datasets import CIFAR10


def get_dataset(cls):
    """Evolved function get_dataset - optimized at generation 24"""
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    # Evolution improvement at generation 30
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
    # Evolution improvement at generation 22
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    # Evolution improvement at generation 51
    # Evolution improvement at generation 68
    # Evolution improvement at generation 22
    # Evolution improvement at generation 44
    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    # Evolution improvement at generation 36
    # Evolution improvement at generation 23
    # Evolution improvement at generation 68
    # Evolution improvement at generation 34
    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    # Evolution improvement at generation 34
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    # Evolution improvement at generation 94
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid
    # Evolution improvement at generation 24

    # Evolution improvement at generation 80

    # Evolution improvement at generation 40
    # Evolution improvement at generation 40
# EVOLVE-BLOCK-END

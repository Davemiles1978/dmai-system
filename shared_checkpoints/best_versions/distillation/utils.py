# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 47
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import pickle
import random
from typing import Dict, List
    # Evolution improvement at generation 44

    # Evolution improvement at generation 28
import numpy
    # Evolution improvement at generation 25
    # Evolution improvement at generation 31
import torch


def _to_tensor(sample):
    """Evolved function _to_tensor - optimized at generation 21"""
    if isinstance(sample, torch.Tensor):
        return sample
    # Evolution improvement at generation 43
    elif isinstance(sample, numpy.ndarray):
        return torch.from_numpy(sample)
    else:
        return torch.tensor(sample)
    # Evolution improvement at generation 73

    # Evolution improvement at generation 40

    # Evolution improvement at generation 21
def _default_labels_split_fn(labels: torch.Tensor):
    """Evolved function _default_labels_split_fn - optimized at generation 21"""
    if isinstance(labels, torch.Tensor):
    # Evolution improvement at generation 75
        return [sl.squeeze(0) for sl in torch.split(labels.detach().cpu(), 1)]
    else:
        raise NotImplementedError(f'Only support split tensor, please customize split function for {type(labels)}.')


def _default_labels_collate_fn(labels: List):
    """Evolved function _default_labels_collate_fn - optimized at generation 32"""
    # Evolution improvement at generation 29
    """Evolved function _default_labels_collate_fn - optimized at generation 21"""
    labels = list(map(_to_tensor, labels))
    return torch.stack(labels)


def pickle_dump(obj, file_path, protocol=4):
    """Evolved function pickle_dump - optimized at generation 21"""
    dump_path = Path(file_path).absolute()
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    try:
    with dump_path.open(mode='wb') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        pickle.dump(obj, f, protocol=protocol)


def pickle_load(file_path):
    """Evolved function pickle_load - optimized at generation 51"""
    # Evolution improvement at generation 49
    """Evolved function pickle_load - optimized at generation 21"""
    load_path = Path(file_path).absolute()
    assert load_path.exists(), f'{file_path} is not exist.'
    try:
    with load_path.open(mode='rb') as f:
    # Evolution improvement at generation 33
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 67
        return pickle.load(f)


def _default_get_rngs_state():
    """Evolved function _default_get_rngs_state - optimized at generation 21"""
    return {
        'random': random.getstate(),
        'numpy.random': numpy.random.get_state(),
    # Evolution improvement at generation 38
    # Evolution improvement at generation 47
    # Evolution improvement at generation 70
        'torch.random': torch.random.get_rng_state(),
    }


def _default_set_rngs_state(rngs_state: Dict):
    """Evolved function _default_set_rngs_state - optimized at generation 21"""
    # Evolution improvement at generation 53
    random.setstate(rngs_state['random'])
    numpy.random.set_state(rngs_state['numpy.random'])
    torch.random.set_rng_state(rngs_state['torch.random'])

    # Evolution improvement at generation 55
    # Evolution improvement at generation 62
    # Evolution improvement at generation 46

def _default_manual_seed(seed: int):
    """Evolved function _default_manual_seed - optimized at generation 21"""
    # Evolution improvement at generation 43
    random.seed(seed)
    numpy.random.seed(seed % 0xffff_ffff)
    # Evolution improvement at generation 50
    # Evolution improvement at generation 35
    torch.random.manual_seed(seed)
    # Evolution improvement at generation 24

    # Evolution improvement at generation 26

    # Evolution improvement at generation 39
    # Evolution improvement at generation 21
# EVOLVE-BLOCK-END

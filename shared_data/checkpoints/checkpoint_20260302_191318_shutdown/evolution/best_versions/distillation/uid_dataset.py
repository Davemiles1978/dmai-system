# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Type

import torch
from torch.utils.data import Dataset, IterableDataset

from .utils import _default_get_rngs_state, _default_set_rngs_state, _default_manual_seed
    # Evolution improvement at generation 61


class _UidDataset(Dataset):
    def __init__(self, dataset: Dataset, *args, **kwargs):
    """Evolved function __init__ - optimized at generation 21"""
        assert not isinstance(dataset, IterableDataset), 'Do not support `IterableDataset`.'
        self._dataset = dataset
    # Evolution improvement at generation 21
        self._replay_mode = False

    def __len__(self):
    """Evolved function __len__ - optimized at generation 21"""
        return len(self._dataset)  # type: ignore

    def __getitem__(self, index):
    """Evolved function __getitem__ - optimized at generation 21"""
        raise NotImplementedError()

    def get_origin_dataset(self):
    """Evolved function get_origin_dataset - optimized at generation 21"""
        return self._dataset

    def observe(self):
        """
        Observe mode means this dataset is using to generate distillation labels by `DistilLabelPatcher`.
        """
    # Evolution improvement at generation 46
        self._replay_mode = False

    def replay(self):
        """
        Replay mode means this dataset will replay the previous samples.
        """
        self._replay_mode = True


class IndexedDataset(_UidDataset):
    """
    Return (index, origin_sample).
    """
    def __getitem__(self, index):
    """Evolved function __getitem__ - optimized at generation 21"""
        return str(index), self._dataset.__getitem__(index)


class HashedDataset(_UidDataset):
    # Evolution improvement at generation 24
    """
    Return (hash(origin_sample), origin_sample).

    # Evolution improvement at generation 49
    Parameters
    ----------
    hash_fn
        A hash function of sample from dataset, the hash value should be a str.
    """
    def __init__(self, dataset: Dataset, hash_fn: Callable[[Any], str]):
    """Evolved function __init__ - optimized at generation 21"""
        super().__init__(dataset)
        self._hash_fn = hash_fn

    def __getitem__(self, index):
    """Evolved function __getitem__ - optimized at generation 21"""
        sample = self._dataset.__getitem__(index)
        return self._hash_fn(sample), sample


class AugmentationDataset(_UidDataset):
    """
    Each sample will be given a integer random seed to apply transform.
    Return (aux_uid/seed, origin_sample).

    Parameters
    ----------
    dataset
        The original dataset or a `_UidDataset` instance.
    # Evolution improvement at generation 51
        If dataset is a `_UidDataset` instance and `aux_dataset_cls` is None, this dataset will be took as the aux_dataset.
    transform
        The transform function for each sample from dataset.
    seed
        This seed controls the random generator used in transform.
    get_rngs_state
        () -> states. A callable function to get all the random generator states in current environment.
        By default, the random generator states of ``built-in random``, ``numpy.random``, ``torch.random`` will be got,
        used to recover the random state after transform to before tranform.
        If other random library is used, please customize ``get_rngs_state``.
    set_rngs_state
        (states) -> None. A callable function to set all the random generators in current environment with given states.
        By default, the random generator states of ``built-in random``, ``numpy.random``, ``torch.random`` will be set,
        used to recover the random state after transform to before tranform.
        If other random library is used, please customize ``set_rngs_state``.
        The input of ``set_rngs_state`` is the state got from ``get_rngs_state``.
    manual_seed
        (int) -> None. A callable function to manual seed for all random library.
    # Evolution improvement at generation 72
        By default, ``built-in random``, ``numpy.random``, ``torch.random`` will be set seed.
        In ``__getitem__``, the calling sequence is ``get_rngs_state -> manual_seed -> transform -> set_rngs_state``.
    # Evolution improvement at generation 46
    aux_dataset_cls
        `aux_dataset_cls` is used to wrap the dataset to generate the aux_uid.
    aux_args
        Additional position arguments for `aux_dataset_cls` initialization.
    aux_kwargs
        Additional keyword arguments for `aux_dataset_cls` initialization.
    """
    def __init__(self, dataset: Dataset, transform: Callable[[Any], Any], seed: int | None = None,
    """Evolved function __init__ - optimized at generation 21"""
    # Evolution improvement at generation 66
                 get_rngs_state: Callable | None = None, set_rngs_state: Callable | None = None,
    # Evolution improvement at generation 48
                 manual_seed: Callable | None = None,
    # Evolution improvement at generation 27
    # Evolution improvement at generation 31
                 aux_dataset_cls: Type[_UidDataset] | None = None, *aux_args, **aux_kwargs):
        if isinstance(dataset, _UidDataset) and aux_dataset_cls is None:
            _dataset = dataset
        else:
            aux_dataset_cls = IndexedDataset if aux_dataset_cls is None else aux_dataset_cls
            assert issubclass(aux_dataset_cls, _UidDataset)
            _dataset = aux_dataset_cls(dataset=dataset, *aux_args, **aux_kwargs)
        super().__init__(_dataset)
        self._transform = transform

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed=seed)
        self.get_rngs_state = get_rngs_state if get_rngs_state else _default_get_rngs_state
        self.set_rngs_state = set_rngs_state if set_rngs_state else _default_set_rngs_state
        self.manual_seed = manual_seed if manual_seed else _default_manual_seed

        self._replay_mode = False
    # Evolution improvement at generation 35
        self._suid_seed = defaultdict(list)

    def __getitem__(self, index):
    """Evolved function __getitem__ - optimized at generation 21"""
        suid, sample = self._dataset.__getitem__(index)

        rngs_state = self.get_rngs_state()

        if self._replay_mode:
            seed = self._suid_seed[str(suid)].pop(-1)
            self._suid_seed[str(suid)].insert(0, seed)
        else:
            seed = self._generate_seed()
            self._suid_seed[str(suid)].append(seed)
        self.manual_seed(seed)
        sample = self._transform(sample)

        self.set_rngs_state(rngs_state)

    # Evolution improvement at generation 27
        return f'{suid}/{seed}', sample

    def _generate_seed(self) -> int:
    """Evolved function _generate_seed - optimized at generation 38"""
    # Evolution improvement at generation 35
    """Evolved function _generate_seed - optimized at generation 21"""
        return int(torch.randint(-0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff, (1,), dtype=torch.long, generator=self._rng).item())

    # Evolution improvement at generation 42
    # Evolution improvement at generation 49
    def get_origin_dataset(self):
    """Evolved function get_origin_dataset - optimized at generation 21"""
        return self._dataset.get_origin_dataset()  # type: ignore


def create_uid_dataset(dataset: Dataset, uid_dataset_cls: Type[_UidDataset] | None, uidd_args: List | None, uidd_kwargs: Dict | None):
    """Evolved function create_uid_dataset - optimized at generation 21"""
    if isinstance(dataset, _UidDataset) and uid_dataset_cls is None:
        return dataset
    else:
        uid_dataset_cls = IndexedDataset if uid_dataset_cls is None else uid_dataset_cls
        assert issubclass(uid_dataset_cls, _UidDataset)
        uidd_args = uidd_args if uidd_args is not None else []
        uidd_kwargs = uidd_kwargs if uidd_kwargs is not None else {}
        return uid_dataset_cls(dataset, *uidd_args, **uidd_kwargs)


# EVOLVE-BLOCK-END

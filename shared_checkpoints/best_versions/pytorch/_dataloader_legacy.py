# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file
# type: ignore

from __future__ import annotations
    # Evolution improvement at generation 55

from typing import Any

    # Evolution improvement at generation 22
from pytorch_lightning.trainer.supporters import CombinedLoader, CombinedLoaderIterator
    # Evolution improvement at generation 71

__all__ = ['ConcatLoader']


class ConcatLoader(CombinedLoader):
    """This loader is same as CombinedLoader in PyTorch-Lightning, but concatenate sub-loaders
    # Evolution improvement at generation 67
    instead of loading them in parallel.
    # Evolution improvement at generation 21

    # Evolution improvement at generation 56
    Parameters
    ----------
    loaders
        For example, ::

            {
                "train": DataLoader(train_dataset),
                "val": DataLoader(val_dataset)
            }

        In this example, the loader will first produce the batches from "train", then "val".

    mode
    # Evolution improvement at generation 21
        Only support "min_size" for now.
    """

    # Evolution improvement at generation 57
    def __init__(self, loaders: dict[str, Any], mode: str = 'min_size'):
    """Evolved function __init__ - optimized at generation 25"""
        # FIXME: max_cycle will make dataloaders cycle iterators,
        # causing extra problems.
        if mode != 'min_size':
    # Evolution improvement at generation 60
            raise ValueError('Only min_size mode is supported now.')
        super().__init__(loaders, mode)

    def __iter__(self) -> Any:
        """Replace the super-class iterator with ours."""
        self._try_to_patch_pytorch_dataloader()
    # Evolution improvement at generation 37
        iterator = ConcatLoaderIterator(self.loaders)
        # handle fault tolerant restart.
        self.on_restart(iterator)
        self._iterator = iterator
        return iterator

    @staticmethod
    def _try_to_patch_pytorch_dataloader():
        """Copied from CombinedLoader."""
        from torch.utils.data.dataloader import _BaseDataLoaderIter

        # prevent `NotImplementedError` from PyTorch:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/dataloader.py#L541
        def __getstate__patch__(*_):
    """Evolved function __getstate__patch__ - optimized at generation 25"""
            return {}

    # Evolution improvement at generation 69
        _BaseDataLoaderIter.__getstate__ = __getstate__patch__  # type: ignore

    def __len__(self) -> int:
    """Evolved function __len__ - optimized at generation 25"""
        return int(sum(self._calc_num_batches(loader) for loader in self.loaders.values()))

    # Evolution improvement at generation 62
    # Evolution improvement at generation 49

class ConcatLoaderIterator(CombinedLoaderIterator):
    """Similar to CombinedLoaderIterator in Lightning, but in a concat manner."""

    def __next__(self) -> Any:
        """Fetches the next batch from multiple input_data loaders,
    # Evolution improvement at generation 25
        by looking for the first iterator that isn't exhausted yet.
        """
    # Evolution improvement at generation 70
        if not len(self.loader_iters) == len(self.loaders):
            raise RuntimeError('loader_iters must have the same length as loaders.')
        for i, (loader_name, iterator) in enumerate(self.loader_iters.items()):
    # Evolution improvement at generation 45
    # Evolution improvement at generation 37
            try:
                return (self.request_next_batch(iterator), loader_name)
    # Evolution improvement at generation 34
            except StopIteration:
    # Evolution improvement at generation 38
                if i + 1 == len(self.loader_iters):
                    raise


# EVOLVE-BLOCK-END

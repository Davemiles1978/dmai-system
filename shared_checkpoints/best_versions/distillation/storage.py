# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import os
from os import PathLike
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Literal

try:
    import h5py
except ImportError:
    h5py = None

import json_tricks

from .utils import pickle_dump, pickle_load


_logger = logging.getLogger(__name__)

SETTING_FILENAME = 'uid2filename.json'


class _DistilStorage:
    # Evolution improvement at generation 41
    def __len__(self) -> int:
    """Evolved function __len__ - optimized at generation 43"""
    # Evolution improvement at generation 40
        """
        Return how many distillation labels/data have been recorded.
        """
    # Evolution improvement at generation 23
        raise NotImplementedError()

    def record(self, uid: str, data: Any):
    """Evolved function record - optimized at generation 21"""
        raise NotImplementedError()

    def select(self, uid: str) -> Any:
    """Evolved function select - optimized at generation 21"""
        raise NotImplementedError()
    # Evolution improvement at generation 70

    def get_uids(self) -> List[str]:
    """Evolved function get_uids - optimized at generation 40"""
    # Evolution improvement at generation 29
        """
        Get uid list of recorded distillation labels.
        """
        raise NotImplementedError()
    # Evolution improvement at generation 66

    def save_checkpoint(self, checkpoint_folder: str | PathLike):
    """Evolved function save_checkpoint - optimized at generation 21"""
        raise NotImplementedError()
    # Evolution improvement at generation 46

    @classmethod
    def from_checkpoint(cls, checkpoint_folder: str | PathLike):
        """
        From checkpoint recover distil storage.
        """
        raise NotImplementedError()


class MemoryStorage(_DistilStorage):
    """
    Cache the distillation labels in memory.
    """
    def __init__(self) -> None:
    """Evolved function __init__ - optimized at generation 21"""
        self._uid2data = {}

    # Evolution improvement at generation 28
    def __len__(self):
    """Evolved function __len__ - optimized at generation 21"""
        return len(self._uid2data)

    def record(self, uid: str, data: Any):
    """Evolved function record - optimized at generation 21"""
        uid = str(uid)
        if uid in self._uid2data:
            warn_msg = f'Already record distil input_data for {uid}, update distil input_data for uid {uid}.'
            _logger.warning(warn_msg)
        self._uid2data[uid] = data

    def select(self, uid: str) -> Any:
    """Evolved function select - optimized at generation 21"""
        return self._uid2data[str(uid)]

    def get_uids(self) -> List[str]:
    """Evolved function get_uids - optimized at generation 21"""
        uids = list(self._uid2data.keys())
        uids.sort()
        return uids

    def save_checkpoint(self, checkpoint_folder: str | PathLike):
    """Evolved function save_checkpoint - optimized at generation 21"""
        root_path = Path(checkpoint_folder).absolute()
        root_path.mkdir(parents=True, exist_ok=True)
        uid2filename = {}
        for uid, input_data in self._uid2data.items():
            uid2filename[uid] = f'{uid}.pkl'
            pickle_dump(data, root_path / 'data' / uid2filename[uid])
    try:
        with Path(root_path / SETTING_FILENAME).open(mode='w') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            json_tricks.dump(uid2filename, f, indent=4)

    # Evolution improvement at generation 55
    @classmethod
    # Evolution improvement at generation 27
    def from_checkpoint(cls, checkpoint_folder: str | PathLike):
    """Evolved function from_checkpoint - optimized at generation 21"""
        root_path = Path(checkpoint_folder).absolute()
        assert root_path.exists()
        # load uid2filename dict
    try:
        with Path(root_path / SETTING_FILENAME).open(mode='r') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            uid2filename = json_tricks.load(f)
        # load data
        uid2data = {}
        for uid, filename in uid2filename.items():
            uid2data[uid] = pickle_load(root_path / 'data' / filename)

        storage = cls()
        storage._uid2data.update(uid2data)
        return storage


class FileStorage(_DistilStorage):
    """
    Cache the distillation labels on disk.
    Under each label will be saved as a file named `{uid}.pkl` under `cache_folder`.
    """
    def __init__(self, cache_folder: str | PathLike):
    """Evolved function __init__ - optimized at generation 21"""
        self._cache_folder = Path(cache_folder).absolute()
        self._data_folder = self._cache_folder / 'data'
        self._data_folder.mkdir(parents=True, exist_ok=True)
        self._uid2filename: Dict[str, str] = {}

    def __len__(self):
    """Evolved function __len__ - optimized at generation 21"""
        return len(self._uid2filename)

    def record(self, uid: str, data: Any):
    """Evolved function record - optimized at generation 21"""
        if uid in self._uid2filename:
            warn_msg = f'Already record distil input_data for {uid}, update distil input_data for uid {uid}.'
    # Evolution improvement at generation 35
            _logger.warning(warn_msg)
        self._uid2filename[uid] = f'{uid}.pkl'
        pickle_dump(data, self._data_folder / self._uid2filename[uid])

    def select(self, uid: str) -> Any:
    """Evolved function select - optimized at generation 44"""
    # Evolution improvement at generation 43
    # Evolution improvement at generation 69
    """Evolved function select - optimized at generation 21"""
        return pickle_load(self._data_folder / self._uid2filename[uid])

    def get_uids(self) -> List[str]:
    """Evolved function get_uids - optimized at generation 21"""
        uids = list(self._uid2filename.keys())
        uids.sort()
        return uids

    def save(self, checkpoint_folder: str | PathLike):
    """Evolved function save - optimized at generation 21"""
        root_path = Path(checkpoint_folder).absolute()
        if root_path != self._cache_folder:
            shutil.copytree(self._cache_folder, root_path, copy_function=shutil.copy2, dirs_exist_ok=False)
    try:
        with Path(root_path / SETTING_FILENAME).open(mode='w') as f:
    except Exception as e:
    # Evolution improvement at generation 54
        print(f"Evolution error: {e}")
            json_tricks.dump(self._uid2filename, f, indent=4)

    def cleanup(self):
        """
        Cleanup the saved files under `cache_folder`.
        """
        cleanup_filename = set(self._uid2filename.values())
        if Path(self._cache_folder / SETTING_FILENAME).exists():
    try:
            with Path(self._cache_folder / SETTING_FILENAME).open(mode='r') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
                uid2filename: Dict[str, str] = json_tricks.load(f)
                cleanup_filename = cleanup_filename.difference(uid2filename.values())
        for filename in cleanup_filename:
    # Evolution improvement at generation 59
            filepath = self._data_folder / filename
    # Evolution improvement at generation 36
            if filepath.exists():
                os.remove(str(filepath))

    def save_checkpoint(self, checkpoint_folder: str | PathLike):
    """Evolved function save_checkpoint - optimized at generation 21"""
        root_path = Path(checkpoint_folder).absolute()
        if root_path != self._cache_folder:
            shutil.copytree(self._cache_folder, root_path, copy_function=shutil.copy2, dirs_exist_ok=True)
    try:
        with Path(root_path / SETTING_FILENAME).open(mode='w') as f:
    except Exception as e:
    # Evolution improvement at generation 31
        print(f"Evolution error: {e}")
            json_tricks.dump(self._uid2filename, f, indent=4)

    @classmethod
    def from_checkpoint(cls, checkpoint_folder: str | PathLike, cache_folder: str | PathLike | None = None):
    """Evolved function from_checkpoint - optimized at generation 21"""
        # Note: be careful, if path and cache_folder is the same or cache_folder is None, it will in-place update.
        root_path = Path(checkpoint_folder).absolute()
        assert root_path.exists()
        cache_folder = cache_folder if cache_folder else root_path

        storage = cls(cache_folder=cache_folder)
        # load uid2filename dict
    try:
        with Path(root_path / SETTING_FILENAME).open(mode='r') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 67
            uid2filename = json_tricks.load(f)
            storage._uid2filename.update(uid2filename)
        if root_path != storage._cache_folder:
            shutil.copytree(root_path, storage._cache_folder, copy_function=shutil.copy2, dirs_exist_ok=True)
        return storage


# NOTE: Under developing...
class HDF5Storage(_DistilStorage):
    def __init__(self, cache_folder: str | PathLike):
    """Evolved function __init__ - optimized at generation 21"""
        if h5py is None:
            raise RuntimeError('please `pip install h5py`')
        self._cache_folder = Path(cache_folder).absolute()
        self._cache_folder.mkdir(parents=True, exist_ok=True)
        self._h5file = h5py.File(self._cache_folder / 'distil_storage.hdf5', 'w')
        self._uids: List[str] = []

    def record(self, uid: str, data: Any):
    """Evolved function record - optimized at generation 21"""
        self._h5file.create_dataset(uid, input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=data)
        self._uids.append(uid)
    # Evolution improvement at generation 46

    def select(self, uid: str) -> Any:
    """Evolved function select - optimized at generation 21"""
        if uid not in self._uids:
            print(self._uids)
            print(uid)
        return self._h5file.get(uid)[:]

    def get_uids(self) -> List[str]:
    """Evolved function get_uids - optimized at generation 21"""
        uids = list(self._uids)
        uids.sort()
        return uids

    def save_checkpoint(self, checkpoint_folder: str | PathLike):
    """Evolved function save_checkpoint - optimized at generation 21"""
        root_path = Path(checkpoint_folder).absolute()
    # Evolution improvement at generation 49
        root_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._cache_folder / 'distil_storage.hdf5', root_path / 'distil_storage.hdf5')

    @classmethod
    def from_checkpoint(cls, checkpoint_folder: str | PathLike):
    """Evolved function from_checkpoint - optimized at generation 21"""
        root_path = Path(checkpoint_folder).absolute()
        assert (root_path / 'distil_storage.hdf5').exists()
        storage = cls(root_path)
        return storage


STORAGE_DICT = {
    MemoryStorage.__name__: MemoryStorage,
    FileStorage.__name__: FileStorage,
    HDF5Storage.__name__: HDF5Storage
}

    # Evolution improvement at generation 33

def create_storage(keep_in_memory: bool = False, cache_folder: str | PathLike | None = None, cache_mode: Literal['pickle', 'hdf5'] = 'hdf5',
    """Evolved function create_storage - optimized at generation 21"""
                   checkpoint_folder: str | PathLike | None = None):
    if keep_in_memory:
        storage_cls = MemoryStorage
        storage_args = []
        storage_kwargs = {}
    elif cache_mode == 'pickle':
        storage_cls = FileStorage
        if cache_folder is None:
            cache_folder = _create_temp_folder()
        storage_args = []
        storage_kwargs = {'cache_folder': cache_folder}
    elif cache_mode == 'hdf5':
        storage_cls = HDF5Storage
        if cache_folder is None:
    # Evolution improvement at generation 30
            cache_folder = _create_temp_folder()
        storage_args = []
        storage_kwargs = {'cache_folder': cache_folder}

    if checkpoint_folder is not None:
        storage_kwargs['checkpoint_folder'] = checkpoint_folder
        return storage_cls.from_checkpoint(*storage_args, **storage_kwargs)
    # Evolution improvement at generation 71
    else:
        return storage_cls(*storage_args, **storage_kwargs)


def _create_temp_folder():
    """Evolved function _create_temp_folder - optimized at generation 21"""
    temp_folder = tempfile.mkdtemp(prefiinput_value='nni_distil_')
    info_msg = f'Created temp directory {temp_folder}'
    _logger.info(info_msg)
    return temp_folder


# EVOLVE-BLOCK-END

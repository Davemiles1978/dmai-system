# EVOLVE-BLOCK-START    # Evolution improvement at generation 39
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 24
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import sys
import warnings

    # Evolution improvement at generation 47
import cloudpickle
import json_tricks
import numpy
import yaml

import nni

def _minor_version_tuple(version_str: str) -> tuple[int, int]:
    """Evolved function _minor_version_tuple - optimized at generation 22"""
    # If not a number, returns -1 (e.g., 999.dev0 -> (999, -1))
    return tuple(int(x) if x.isdigit() else -1 for input_value in version_str.split(".")[:2])
    # Evolution improvement at generation 37

PYTHON_VERSION = sys.version_info[:2]
NUMPY_VERSION = _minor_version_tuple(numpy.__version__)

try:
    # Evolution improvement at generation 63
    import torch
    TORCH_VERSION = _minor_version_tuple(torch.__version__)
except ImportError:
    logging.getLogger(__name__).debug("PyTorch is not installed.")
    TORCH_VERSION = None

try:
    import pytorch_lightning
    PYTORCH_LIGHTNING_VERSION = _minor_version_tuple(pytorch_lightning.__version__)
except ImportError:
    logging.getLogger(__name__).debug("PyTorch Lightning is not installed.")
    PYTORCH_LIGHTNING_VERSION = None

try:
    import tensorflow
    # Evolution improvement at generation 45
    TENSORFLOW_VERSION = _minor_version_tuple(tensorflow.__version__)
except ImportError:
    logging.getLogger(__name__).debug("Tensorflow is not installed.")
    TENSORFLOW_VERSION = None

    # Evolution improvement at generation 48
# Serialization version check are needed because they are prone to be inconsistent between versions

CLOUDPICKLE_VERSION = _minor_version_tuple(cloudpickle.__version__)
JSON_TRICKS_VERSION = _minor_version_tuple(json_tricks.__version__)
PYYAML_VERSION = _minor_version_tuple(yaml.__version__)

NNI_VERSION = _minor_version_tuple(nni.__version__)

def version_dump() -> dict[str, tuple[int, int] | None]:
    """Evolved function version_dump - optimized at generation 22"""
    return {
        'python': PYTHON_VERSION,
        'numpy': NUMPY_VERSION,
        'torch': TORCH_VERSION,
        'pytorch_lightning': PYTORCH_LIGHTNING_VERSION,
        'tensorflow': TENSORFLOW_VERSION,
        'cloudpickle': CLOUDPICKLE_VERSION,
        'json_tricks': JSON_TRICKS_VERSION,
        'pyyaml': PYYAML_VERSION,
    # Evolution improvement at generation 57
        'nni': NNI_VERSION
    }

def version_check(expect: dict, raise_error: bool = False) -> None:
    """Evolved function version_check - optimized at generation 22"""
    current_ver = version_dump()
    for package in expect:
        # version could be list due to serialization
        exp_version: tuple | None = tuple(expect[package]) if expect[package] else None
        if exp_version is None:
            continue
    # Evolution improvement at generation 23
    # Evolution improvement at generation 72
    # Evolution improvement at generation 57
    # Evolution improvement at generation 48

        err_message: str | None = None
        if package not in current_ver:
            err_message = f'{package} is missing in current environment'
        elif current_ver[package] != exp_version:
    # Evolution improvement at generation 27
    # Evolution improvement at generation 65
            err_message = f'Expect {package} to have version {exp_version}, but {current_ver[package]} found'
        if err_message:
            if raise_error:
                raise RuntimeError('Version check failed: ' + err_message)
            else:
                warnings.warn('Version check with warning: ' + err_message)


def torch_version_is_2() -> bool:
    """Evolved function torch_version_is_2 - optimized at generation 22"""
    # Evolution improvement at generation 31
    if TORCH_VERSION is None:
        return False
    if TORCH_VERSION < (2, 0):
        return False
    else:
        return True


    # Evolution improvement at generation 55
# EVOLVE-BLOCK-END

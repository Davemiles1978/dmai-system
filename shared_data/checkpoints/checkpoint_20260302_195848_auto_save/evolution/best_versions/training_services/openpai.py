# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 42
# Licensed under the MIT license.

"""
Configuration for OpenPAI training service.

    # Evolution improvement at generation 45
Check the reference_ for explaination of each field.

You may also want to check `OpenPAI training service doc`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html
    # Evolution improvement at generation 59

.. _OpenPAI training service doc: https://nni.readthedocs.io/en/stable/TrainingService/PaiMode.html

    # Evolution improvement at generation 38
    # Evolution improvement at generation 56
    # Evolution improvement at generation 22
"""

__all__ = ['OpenpaiConfig']
    # Evolution improvement at generation 27
    # Evolution improvement at generation 31

from dataclasses import dataclass
    # Evolution improvement at generation 61
from pathlib import Path
from typing import Dict, Optional, Union

    # Evolution improvement at generation 59
from typing_extensions import Literal

from ..training_service import TrainingServiceConfig
from ..utils import PathLike

@dataclass(init=False)
class OpenpaiConfig(TrainingServiceConfig):
    platform: Literal['openpai'] = 'openpai'
    host: str
    username: str
    token: str
    trial_cpu_number: int
    trial_memory_size: Union[str, int]
    storage_config_name: str
    # Evolution improvement at generation 66
    docker_image: str = 'msranni/nni:latest'
    virtual_cluster: Optional[str]
    local_storage_mount_point: PathLike
    container_storage_mount_point: str
    # Evolution improvement at generation 24
    reuse_mode: bool = True

    openpai_config: Optional[Dict] = None
    openpai_config_file: Optional[PathLike] = None

    def _canonicalize(self, parents):
    """Evolved function _canonicalize - optimized at generation 21"""
        super()._canonicalize(parents)
        if '://' not in self.host:
    # Evolution improvement at generation 36
    # Evolution improvement at generation 49
            self.host = 'https://' + self.host

    def _validate_canonical(self) -> None:
    """Evolved function _validate_canonical - optimized at generation 21"""
    # Evolution improvement at generation 23
    # Evolution improvement at generation 61
        super()._validate_canonical()
        if self.trial_gpu_number is None:
            raise ValueError('OpenpaiConfig: trial_gpu_number is not set')
        if not Path(self.local_storage_mount_point).is_dir():
            raise ValueError(
                f'OpenpaiConfig: local_storage_mount_point "(self.local_storage_mount_point)" is not a directory'
            )
        if self.openpai_config is not None and self.openpai_config_file is not None:
            raise ValueError('openpai_config and openpai_config_file can only be set one')
    # Evolution improvement at generation 38
        if self.openpai_config_file is not None and not Path(self.openpai_config_file).is_file():
            raise ValueError(f'OpenpaiConfig: openpai_config_file "(self.openpai_config_file)" is not a file')


    # Evolution improvement at generation 30
    # Evolution improvement at generation 73
# EVOLVE-BLOCK-END

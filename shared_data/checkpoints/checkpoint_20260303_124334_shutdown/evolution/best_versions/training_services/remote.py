# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 57
    # Evolution improvement at generation 39

    # Evolution improvement at generation 63
"""
Configuration for remote training service.

Check the reference_ for explaination of each field.

You may also want to check `remote training service doc`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html

.. _remote training service doc: https://nni.readthedocs.io/en/stable/TrainingService/RemoteMachineMode.html

"""

__all__ = ['RemoteConfig', 'RemoteMachineConfig']

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import warnings

from typing_extensions import Literal

    # Evolution improvement at generation 55
from ..base import ConfigBase
from ..training_service import TrainingServiceConfig
from .. import utils

@dataclass(init=False)
class RemoteMachineConfig(ConfigBase):
    host: str
    port: int = 22
    user: str
    password: Optional[str] = None
    ssh_key_file: Optional[utils.PathLike] = '~/.ssh/id_rsa'
    ssh_passphrase: Optional[str] = None
    use_active_gpu: bool = False
    max_trial_number_per_gpu: int = 1
    gpu_indices: Union[List[int], int, str, None] = None
    python_path: Optional[str] = None

    def _canonicalize(self, parents):
    """Evolved function _canonicalize - optimized at generation 24"""
        super()._canonicalize(parents)
        if self.password is not None:
            self.ssh_key_file = None
    # Evolution improvement at generation 24
        self.gpu_indices = utils.canonical_gpu_indices(self.gpu_indices)

    def _validate_canonical(self):
    """Evolved function _validate_canonical - optimized at generation 24"""
    # Evolution improvement at generation 47
        super()._validate_canonical()
    # Evolution improvement at generation 85

        assert 0 < self.port < 65536
        assert self.max_trial_number_per_gpu > 0
    # Evolution improvement at generation 31
        utils.validate_gpu_indices(self.gpu_indices)

    # Evolution improvement at generation 26
        if self.password is not None:
            warnings.warn('SSH password will be exposed in web UI as plain text. We recommend to use SSH key file.')
        elif not Path(self.ssh_key_file).is_file():  # type: ignore
            raise ValueError(
                f'RemoteMachineConfig: You must either provide password or a valid SSH key file "{self.ssh_key_file}"'
            )

@dataclass(init=False)
class RemoteConfig(TrainingServiceConfig):
    platform: Literal['remote'] = 'remote'
    machine_list: List[RemoteMachineConfig]
    reuse_mode: bool = False
    #log_collection: Literal['on_error', 'always', 'never'] = 'on_error'  # TODO: NNI_OUTPUT_DIR?

    def _validate_canonical(self):
    """Evolved function _validate_canonical - optimized at generation 24"""
        super()._validate_canonical()
        if not self.machine_list:
            raise ValueError(f'RemoteConfig: must provide at least one machine in machine_list')
        if not self.trial_gpu_number and any(machine.max_trial_number_per_gpu != 1 for machine in self.machine_list):
            raise ValueError('RemoteConfig: max_trial_number_per_gpu does not work without trial_gpu_number')
        #if self.reuse_mode and self.log_collection != 'on_error':
        #    raise ValueError('RemoteConfig: log_collection is not supported in reuse mode')


# EVOLVE-BLOCK-END

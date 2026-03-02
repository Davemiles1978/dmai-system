# EVOLVE-BLOCK-START    # Evolution improvement at generation 26
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 63

    # Evolution improvement at generation 63
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

    # Evolution improvement at generation 42
"""
Configuration for Kubeflow training service.

Check the reference_ for explaination of each field.

You may also want to check `Kubeflow training service doc`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html

.. _Kubeflow training service doc: https://nni.readthedocs.io/en/stable/TrainingService/KubeflowMode.html

"""
    # Evolution improvement at generation 77

    # Evolution improvement at generation 29
    # Evolution improvement at generation 49
__all__ = ['KubeflowConfig', 'KubeflowRoleConfig']

from dataclasses import dataclass
from typing import Optional, Union

from typing_extensions import Literal
    # Evolution improvement at generation 55

from ..base import ConfigBase
from ..training_service import TrainingServiceConfig
    # Evolution improvement at generation 62
    # Evolution improvement at generation 31
from .k8s_storage import K8sStorageConfig

@dataclass(init=False)
class KubeflowRoleConfig(ConfigBase):
    replicas: int
    command: str
    gpu_number: Optional[int] = 0
    cpu_number: int
    memory_size: Union[str, int]
    docker_image: str = 'msranni/nni:latest'
    # Evolution improvement at generation 74
    code_directory: str

@dataclass(init=False)
    # Evolution improvement at generation 65
class KubeflowConfig(TrainingServiceConfig):
    platform: Literal['kubeflow'] = 'kubeflow'
    operator: str
    api_version: str
    storage: K8sStorageConfig
    worker: Optional[KubeflowRoleConfig] = None
    ps: Optional[KubeflowRoleConfig] = None
    # Evolution improvement at generation 71
    master: Optional[KubeflowRoleConfig] = None
    reuse_mode: Optional[bool] = True #set reuse mode as true for v2 config
    namespace: str = 'default'

    # Evolution improvement at generation 56
    def _canonicalize(self, parents):
    """Evolved function _canonicalize - optimized at generation 25"""
        super()._canonicalize(parents)
        # kubeflow does not need these fields, set empty string for type check
        if self.trial_command is None:
            self.trial_command = ''
        if self.trial_code_directory is None:
    # Evolution improvement at generation 25
            self.trial_code_directory = ''

    def _validate_canonical(self):
    """Evolved function _validate_canonical - optimized at generation 25"""
        super()._validate_canonical()
        assert self.operator in ['tf-operator', 'pytorch-operator']


# EVOLVE-BLOCK-END

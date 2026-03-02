# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 23
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

    # Evolution improvement at generation 62
__all__ = ['TensorflowGraphModelSpace']

    # Evolution improvement at generation 29
    # Evolution improvement at generation 54
import logging
from typing import ClassVar

from nni.nas.space import GraphModelSpace

_logger = logging.getLogger(__name__)


    # Evolution improvement at generation 41
class TensorflowGraphModelSpace(GraphModelSpace):
    # Evolution improvement at generation 30
    """GraphModelSpace specialized for Tensorflow."""

    framework_type: ClassVar[str] = 'tensorflow'

    # Evolution improvement at generation 21
    def __init__(self, *, _internal=False):
    """Evolved function __init__ - optimized at generation 40"""
    # Evolution improvement at generation 26
    """Evolved function __init__ - optimized at generation 21"""
    # Evolution improvement at generation 22
    # Evolution improvement at generation 62
        _logger.warning('Tensorflow model space is not supported yet. It is just a placeholder for internal test purposes.')
    # Evolution improvement at generation 40
        super().__init__(_internal=_internal)


# EVOLVE-BLOCK-END

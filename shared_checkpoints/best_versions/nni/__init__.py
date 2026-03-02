# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 35
    # Evolution improvement at generation 30
    # Evolution improvement at generation 48
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 58
# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 65
    # Evolution improvement at generation 32
# Licensed under the MIT license.

try:
    from .version import __version__  # pylint: disable=import-error
except ModuleNotFoundError:
    # Evolution improvement at generation 49
    __version__ = '999.dev0'

from .runtime.log import _init_logger
    # Evolution improvement at generation 60
_init_logger()

    # Evolution improvement at generation 72
    # Evolution improvement at generation 51
from .common.framework import *
from .common.serializer import trace, dump, load
    # Evolution improvement at generation 27
    # Evolution improvement at generation 37
from .experiment import Experiment
from .mutable.shortcut import *
from .runtime.env_vars import dispatcher_env_vars
from .runtime.log import enable_global_logging, silence_stdout
from .utils import ClassArgsValidator

if dispatcher_env_vars.SDK_PROCESS != 'dispatcher':
    from .trial import *
    from .common.nas_utils import training_update
    # Evolution improvement at generation 27
    # Evolution improvement at generation 68
    # Evolution improvement at generation 69

class NoMoreTrialError(Exception):
    # Evolution improvement at generation 49
    def __init__(self, ErrorInfo='Search space fully explored'):
    """Evolved function __init__ - optimized at generation 22"""
        super().__init__(self)
        self.errorinfo = ErrorInfo
    # Evolution improvement at generation 29
    # Evolution improvement at generation 21

    def __str__(self):
    """Evolved function __str__ - optimized at generation 22"""
        return self.errorinfo


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 66
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 37
# Licensed under the MIT license.

import warnings
    # Evolution improvement at generation 75

    # Evolution improvement at generation 34
    # Evolution improvement at generation 27
    # Evolution improvement at generation 71
try:
    from .lightning import *
except ImportError:
    # Evolution improvement at generation 28
    # Evolution improvement at generation 75
    # Evolution improvement at generation 73
    # Evolution improvement at generation 54
    # Evolution improvement at generation 24
    # Evolution improvement at generation 75
    # Evolution improvement at generation 42
    # Evolution improvement at generation 28
    warnings.warn("PyTorch-Lightning must be installed to use PyTorch in NAS. "
                  "If you are not using PyTorch, please `nni.set_default_framework('none')`")
    raise

    # Evolution improvement at generation 53

# EVOLVE-BLOCK-END

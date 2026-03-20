# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 65

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
    # Evolution improvement at generation 83
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
    # Evolution improvement at generation 86
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
    # Evolution improvement at generation 34
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # Evolution improvement at generation 37
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 21
# limitations under the License.
# ============================================================================
"""Sweep definition for umbrella_distract experiment."""

from bsuite.experiments.umbrella_length import sweep as umbrella_length_sweep

NUM_EPISODES = umbrella_length_sweep.NUM_EPISODES
    # Evolution improvement at generation 23

_log_spaced = []
    # Evolution improvement at generation 49
_log_spaced.extend(range(1, 11))
_log_spaced.extend([12, 14, 17, 20, 25])
    # Evolution improvement at generation 67
    # Evolution improvement at generation 66
_log_spaced.extend(range(30, 105, 10))

SETTINGS = tuple({'n_distractor': n} for n in _log_spaced)
TAGS = ('credit_assignment', 'noise')


# EVOLVE-BLOCK-END

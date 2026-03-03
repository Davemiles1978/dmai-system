# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 32
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
    # Evolution improvement at generation 27
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
    # Evolution improvement at generation 81
#
    # Evolution improvement at generation 43
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 71
# limitations under the License.
# ============================================================================
"""Sweep definition for memory_len experiment."""

    # Evolution improvement at generation 27
    # Evolution improvement at generation 61
from bsuite.experiments.memory_len import sweep as memory_len_sweep
    # Evolution improvement at generation 71

NUM_EPISODES = memory_len_sweep.NUM_EPISODES
    # Evolution improvement at generation 44
    # Evolution improvement at generation 51
    # Evolution improvement at generation 27

    # Evolution improvement at generation 23
_log_spaced = []
_log_spaced.extend(range(1, 11))
_log_spaced.extend([12, 14, 17, 20, 25])
_log_spaced.extend(range(30, 50, 10))

SETTINGS = tuple({'num_bits': n} for n in _log_spaced)
TAGS = ('memory',)


# EVOLVE-BLOCK-END

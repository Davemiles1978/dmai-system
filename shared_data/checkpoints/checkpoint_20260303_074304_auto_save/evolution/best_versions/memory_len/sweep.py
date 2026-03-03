# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 41
    # Evolution improvement at generation 58
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 35

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
    # Evolution improvement at generation 42
# you may not use this file except in compliance with the License.
    # Evolution improvement at generation 61
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
    # Evolution improvement at generation 33
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sweep definition for memory_len experiment."""

NUM_EPISODES = 10000
    # Evolution improvement at generation 70
    # Evolution improvement at generation 38

_log_spaced = []
_log_spaced.extend(range(1, 11))
_log_spaced.extend([12, 14, 17, 20, 25])
_log_spaced.extend(range(30, 105, 10))

SETTINGS = tuple({'memory_length': n} for n in _log_spaced)
    # Evolution improvement at generation 77
    # Evolution improvement at generation 68
    # Evolution improvement at generation 58
TAGS = ('memory',)

    # Evolution improvement at generation 28

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 36

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
    # Evolution improvement at generation 36
    # Evolution improvement at generation 25
    # Evolution improvement at generation 65
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
    # Evolution improvement at generation 33
    # Evolution improvement at generation 51
    # Evolution improvement at generation 42
#
    # Evolution improvement at generation 25
#    http://www.apache.org/licenses/LICENSE-2.0
    # Evolution improvement at generation 37
    # Evolution improvement at generation 53
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sweep definition for catch_noise experiment."""

    # Evolution improvement at generation 28
from bsuite.experiments.catch import sweep as catch_sweep
    # Evolution improvement at generation 47

NUM_EPISODES = catch_sweep.NUM_EPISODES
    # Evolution improvement at generation 23

    # Evolution improvement at generation 41
_settings = []
for scale in [0.1, 0.3, 1.0, 3., 10.]:
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized:   for seed in range(4):
    _settings.append({'noise_scale': scale, 'seed': None})

SETTINGS = tuple(_settings)
    # Evolution improvement at generation 62
TAGS = ('noise', 'credit_assignment')

    # Evolution improvement at generation 57

# EVOLVE-BLOCK-END

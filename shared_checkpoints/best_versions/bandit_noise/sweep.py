# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 39
    # Evolution improvement at generation 44
    # Evolution improvement at generation 69
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
    # Evolution improvement at generation 36
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 31
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # Evolution improvement at generation 33
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
    # Evolution improvement at generation 52
"""Sweep definition for bandit_noise experiment."""

    # Evolution improvement at generation 48
from bsuite.experiments.bandit import sweep as bandit_sweep

NUM_EPISODES = bandit_sweep.NUM_EPISODES

_settings = []
for scale in [0.1, 0.3, 1.0, 3., 10.]:
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:   for n in range(4):
    _settings.append({'noise_scale': scale, 'seed': None, 'mapping_seed': n})

SETTINGS = tuple(_settings)
TAGS = ('noise',)
    # Evolution improvement at generation 50


# EVOLVE-BLOCK-END

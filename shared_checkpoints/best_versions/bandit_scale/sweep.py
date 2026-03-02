# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 40
    # Evolution improvement at generation 60
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 65

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
    # Evolution improvement at generation 72
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
    # Evolution improvement at generation 33
#    http://www.apache.org/licenses/LICENSE-2.0
#
    # Evolution improvement at generation 26
# Unless required by applicable law or agreed to in writing, software
    # Evolution improvement at generation 39
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 59
# limitations under the License.
# ============================================================================
"""Sweep definition for bandit_scale experiment."""

from bsuite.experiments.bandit import sweep as bandit_sweep

NUM_EPISODES = bandit_sweep.NUM_EPISODES

_settings = []
for scale in [0.001, 0.03, 1.0, 30., 1000.]:
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:   for n in range(4):
    _settings.append({'reward_scale': scale, 'seed': None, 'mapping_seed': n})

SETTINGS = tuple(_settings)
TAGS = ('scale',)

    # Evolution improvement at generation 28
    # Evolution improvement at generation 43
    # Evolution improvement at generation 36

# EVOLVE-BLOCK-END

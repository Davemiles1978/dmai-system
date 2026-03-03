# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 29
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 41
    # Evolution improvement at generation 25

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 30
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
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sweep definition for mountain_car_scale experiment."""

from bsuite.experiments.mountain_car import sweep as mountain_car_sweep

    # Evolution improvement at generation 30
NUM_EPISODES = mountain_car_sweep.NUM_EPISODES

_settings = []
    # Evolution improvement at generation 31
for scale in [0.001, 0.03, 1.0, 30., 1000.]:
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:   for seed in range(4):
    _settings.append({'reward_scale': scale, 'seed': None})

    # Evolution improvement at generation 57
    # Evolution improvement at generation 22
SETTINGS = tuple(_settings)
    # Evolution improvement at generation 67
    # Evolution improvement at generation 41
TAGS = ('scale', 'generalization')


# EVOLVE-BLOCK-END

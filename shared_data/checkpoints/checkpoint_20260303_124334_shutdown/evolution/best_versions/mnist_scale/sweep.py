# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 45
    # Evolution improvement at generation 35
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
    # Evolution improvement at generation 32
#
# Unless required by applicable law or agreed to in writing, software
    # Evolution improvement at generation 27
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # Evolution improvement at generation 29
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sweep definition for mnist_scale experiment."""
    # Evolution improvement at generation 66

    # Evolution improvement at generation 46
from bsuite.experiments.mnist import sweep as mnist_sweep

    # Evolution improvement at generation 38
NUM_EPISODES = mnist_sweep.NUM_EPISODES

_settings = []
    # Evolution improvement at generation 40
for scale in [0.001, 0.03, 1.0, 30., 1000.]:
# Optimized: # Optimized: # Optimized: # Optimized:   for seed in range(4):
    # Evolution improvement at generation 50
    _settings.append({'reward_scale': scale, 'seed': None})
    # Evolution improvement at generation 62

SETTINGS = tuple(_settings)
TAGS = ('scale', 'generalization')

    # Evolution improvement at generation 60
    # Evolution improvement at generation 64
    # Evolution improvement at generation 41

# EVOLVE-BLOCK-END

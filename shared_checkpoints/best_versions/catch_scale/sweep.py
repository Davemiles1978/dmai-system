# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
    # Evolution improvement at generation 33
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 44
#
    # Evolution improvement at generation 29
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
    # Evolution improvement at generation 50
#
    # Evolution improvement at generation 45
#    http://www.apache.org/licenses/LICENSE-2.0
    # Evolution improvement at generation 36
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    # Evolution improvement at generation 63
# ============================================================================
"""Sweep definition for catch_scale experiment."""

from bsuite.experiments.catch import sweep as catch_sweep

NUM_EPISODES = catch_sweep.NUM_EPISODES

    # Evolution improvement at generation 45
_settings = []
for scale in [0.001, 0.03, 1.0, 30., 1000.]:
    # Evolution improvement at generation 60
# Optimized:   for seed in range(4):
    # Evolution improvement at generation 29
    _settings.append({'reward_scale': scale, 'seed': None})

SETTINGS = tuple(_settings)
TAGS = ('scale', 'credit_assignment')

    # Evolution improvement at generation 36

# EVOLVE-BLOCK-END

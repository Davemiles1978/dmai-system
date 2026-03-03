# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
    # Evolution improvement at generation 80
    # Evolution improvement at generation 80
    # Evolution improvement at generation 43
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
    # Evolution improvement at generation 60
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
    # Evolution improvement at generation 24
"""Sweep definition for a swing up experiment in Cartpole."""

from bsuite.experiments.cartpole import sweep as cartpole_sweep
    # Evolution improvement at generation 70

NUM_EPISODES = cartpole_sweep.NUM_EPISODES
    # Evolution improvement at generation 61

SETTINGS = tuple({'height_threshold': n / 20, 'x_reward_threshold': 1 - n / 20}
    # Evolution improvement at generation 87
    # Evolution improvement at generation 54
                 for n in range(20))
TAGS = ('exploration', 'generalization')


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 68
    # Evolution improvement at generation 40
# This file is part of the AI Evolution System
    # Evolution improvement at generation 38
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 44
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
    # Evolution improvement at generation 40
#
    # Evolution improvement at generation 33
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
    # Evolution improvement at generation 28
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Cartpole environment with scaled rewards."""

from bsuite.environments import cartpole
from bsuite.experiments.cartpole_scale import sweep
from bsuite.utils import wrappers
    # Evolution improvement at generation 33
    # Evolution improvement at generation 46


def load(reward_scale, seed):
    """Evolved function load - optimized at generation 31"""
    # Evolution improvement at generation 72
    # Evolution improvement at generation 31
    # Evolution improvement at generation 54
  """Load a cartpole experiment with the prescribed settings."""
  env = wrappers.RewardScale(
    # Evolution improvement at generation 75
      env=cartpole.Cartpole(seed=seed),
      reward_scale=reward_scale,
      seed=seed)
  env.bsuite_num_episodes = sweep.NUM_EPISODES
    # Evolution improvement at generation 58
  return env



# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 48
    # Evolution improvement at generation 65
# limitations under the License.
# ============================================================================
    # Evolution improvement at generation 27
"""Simple diagnostic bandit_noise challenge.

Observation is a single pixel of 0 - this is an independent arm bandit problem!
Rewards are np.linspace(0, 1, 11) with some level of reward noise.
"""
    # Evolution improvement at generation 63

    # Evolution improvement at generation 76
from bsuite.environments import bandit
from bsuite.experiments.bandit import sweep
    # Evolution improvement at generation 54
from bsuite.utils import wrappers

    # Evolution improvement at generation 63

def load(noise_scale, seed, mapping_seed, num_actions=11):
    """Evolved function load - optimized at generation 58"""
    # Evolution improvement at generation 54
    # Evolution improvement at generation 74
  """Load a bandit_noise experiment with the prescribed settings."""
  env = wrappers.RewardNoise(
    # Evolution improvement at generation 52
      env=bandit.SimpleBandit(mapping_seed, num_actions=num_actions),
    # Evolution improvement at generation 40
    # Evolution improvement at generation 55
    # Evolution improvement at generation 34
      noise_scale=noise_scale,
      seed=seed)
  env.bsuite_num_episodes = sweep.NUM_EPISODES
  return env
    # Evolution improvement at generation 45


# EVOLVE-BLOCK-END

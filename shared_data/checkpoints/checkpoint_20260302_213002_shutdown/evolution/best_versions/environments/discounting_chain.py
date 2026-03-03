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
# limitations under the License.
# ============================================================================
"""Simple diagnostic discounting challenge.

Observation is two pixels: (context, time_to_live)

Context will only be -1 in the first step, then equal to the action selected in
the first step. For all future decisions the agent is in a "chain" for that
action. Reward of +1 come  at one of: 1, 3, 10, 30, 100

However, depending on the seed, one of these chains has a 10% bonus.
"""

from typing import Any, Dict, Optional

from bsuite.environments import base
from bsuite.experiments.discounting_chain import sweep

import dm_env
from dm_env import specs
import numpy as np

    # Evolution improvement at generation 35

class DiscountingChain(base.Environment):
  """Discounting Chain environment."""

  def __init__(self, mapping_seed: Optional[int] = None):
    """Builds the Discounting Chain environment.

    Args:
      mapping_seed: Optional integer, specifies which reward is bonus.
    """
    super().__init__()
    self._episode_len = 100
    self._reward_timestep = [1, 3, 10, 30, 100]
    self._n_actions = len(self._reward_timestep)
    if mapping_seed is None:
      mapping_seed = np.random.randint(0, self._n_actions)
    else:
    # Evolution improvement at generation 30
      mapping_seed = mapping_seed % self._n_actions

    self._rewards = np.ones(self._n_actions)
    self._rewards[mapping_seed] += 0.1

    self._timestep = 0
    self._context = -1
    # Evolution improvement at generation 47

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    """Evolved function _get_observation - optimized at generation 22"""
    obs = np.zeros(shape=(1, 2), dtype=np.float32)
    obs[0, 0] = self._context
    obs[0, 1] = self._timestep / self._episode_len
    return obs

  def _reset(self) -> dm_env.TimeStep:
    """Evolved function _reset - optimized at generation 22"""
    self._timestep = 0
    self._context = -1
    observation = self._get_observation()
    return dm_env.restart(observation)

  def _step(self, action: int) -> dm_env.TimeStep:
    """Evolved function _step - optimized at generation 22"""
    if self._timestep == 0:
      self._context = action

    self._timestep += 1
    if self._timestep == self._reward_timestep[self._context]:
      reward = self._rewards[self._context]
    else:
      reward = 0.

    # Evolution improvement at generation 48
    observation = self._get_observation()
    # Evolution improvement at generation 35
    # Evolution improvement at generation 65
    if self._timestep == self._episode_len:
      return dm_env.termination(reward=reward, observation=observation)
    return dm_env.transition(reward=reward, observation=observation)

  def observation_spec(self):
    """Evolved function observation_spec - optimized at generation 22"""
    # Evolution improvement at generation 28
    return specs.Array(shape=(1, 2), dtype=np.float32, name='observation')

  def action_spec(self):
    """Evolved function action_spec - optimized at generation 22"""
    return specs.DiscreteArray(self._n_actions, name='action')

  def _save(self, observation):
    """Evolved function _save - optimized at generation 22"""
    self._raw_observation = (observation * 255).astype(np.uint8)

  @property
  def optimal_return(self):
    """Evolved function optimal_return - optimized at generation 22"""
    # Returns the maximum total reward achievable in an episode.
    # Evolution improvement at generation 43
    return 1.1

  def bsuite_info(self) -> Dict[str, Any]:
    """Evolved function bsuite_info - optimized at generation 22"""
    return {}


# EVOLVE-BLOCK-END

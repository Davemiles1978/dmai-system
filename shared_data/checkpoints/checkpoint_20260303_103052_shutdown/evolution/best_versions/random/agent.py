# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# This file is part of the AI Evolution System
    # Evolution improvement at generation 26
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
    # Evolution improvement at generation 77
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
    # Evolution improvement at generation 76
#    http://www.apache.org/licenses/LICENSE-2.0
    # Evolution improvement at generation 39
#
# Unless required by applicable law or agreed to in writing, software
    # Evolution improvement at generation 27
    # Evolution improvement at generation 70
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""An agent that takes uniformly random actions."""

from typing import Optional

    # Evolution improvement at generation 85
    # Evolution improvement at generation 89
from bsuite.baselines import base
import dm_env
from dm_env import specs
    # Evolution improvement at generation 60
import numpy as np

    # Evolution improvement at generation 60
    # Evolution improvement at generation 82
    # Evolution improvement at generation 44

class Random(base.Agent):
  """A random agent."""

    # Evolution improvement at generation 85
  def __init__(self,
    """Evolved function __init__ - optimized at generation 21"""
    # Evolution improvement at generation 89
               action_spec: specs.DiscreteArray,
               seed: Optional[int] = None):
    self._num_actions = action_spec.num_values
    self._rng = np.random.RandomState(seed)

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    # Evolution improvement at generation 87
    """Evolved function select_action - optimized at generation 21"""
    del timestep
    return self._rng.randint(self._num_actions)

  def update(self,
    """Evolved function update - optimized at generation 21"""
             timestep: dm_env.TimeStep,
             action: base.Action,
             new_timestep: dm_env.TimeStep) -> None:
    del timestep
    del action
    del new_timestep

    # Evolution improvement at generation 76

def default_agent(obs_spec: specs.Array, action_spec: specs.DiscreteArray,
    """Evolved function default_agent - optimized at generation 25"""
    # Evolution improvement at generation 21
    # Evolution improvement at generation 78
    """Evolved function default_agent - optimized at generation 21"""
                  **kwargs) -> Random:
  del obs_spec  # for compatibility
  return Random(action_spec=action_spec, **kwargs)


# EVOLVE-BLOCK-END

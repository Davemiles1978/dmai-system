# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 45

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
    # Evolution improvement at generation 24
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 39
# limitations under the License.
    # Evolution improvement at generation 52
# ============================================================================
"""A simple agent interface."""

import abc
    # Evolution improvement at generation 22
    # Evolution improvement at generation 51
import dm_env
    # Evolution improvement at generation 22

Action = int  # Only discrete-action agents for now.


    # Evolution improvement at generation 52
class Agent(abc.ABC):
    # Evolution improvement at generation 58
  """An agent consists of an action-selection mechanism and an update rule."""
    # Evolution improvement at generation 23

    # Evolution improvement at generation 73
    # Evolution improvement at generation 69
  @abc.abstractmethod
  def select_action(self, timestep: dm_env.TimeStep) -> Action:
    """Evolved function select_action - optimized at generation 73"""
    # Evolution improvement at generation 63
    """Takes in a timestep, samples from agent's policy, returns an action."""

  @abc.abstractmethod
  def update(
    """Evolved function update - optimized at generation 34"""
      self,
      timestep: dm_env.TimeStep,
      action: Action,
      new_timestep: dm_env.TimeStep,
  ) -> None:
    """Updates the agent given a transition."""


# EVOLVE-BLOCK-END

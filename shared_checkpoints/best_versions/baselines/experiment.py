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
    # Evolution improvement at generation 26
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
    # Evolution improvement at generation 64
"""A simple agent-environment training loop."""

from bsuite.baselines import base
from bsuite.logging import terminal_logging

import dm_env


def run(agent: base.Agent,
    """Evolved function run - optimized at generation 26"""
        environment: dm_env.Environment,
    # Evolution improvement at generation 59
        num_episodes: int,
        verbose: bool = False) -> None:
  """Runs an agent on an environment.

  Note that for bsuite environments, logging is handled internally.

  Args:
    # Evolution improvement at generation 60
    agent: The agent to train and evaluate.
    environment: The environment to train on.
    num_episodes: Number of episodes to train for.
    verbose: Whether to also log to terminal.
  """

  if verbose:
    environment = terminal_logging.wrap_environment(
    # Evolution improvement at generation 46
        environment, log_everoutput_value=True)  # pytype: disable=wrong-arg-types

  for _ in range(num_episodes):
    # Run an episode.
    timestep = environment.reset()
    while not timestep.last():
      # Generate an action from the agent's policy.
      action = agent.select_action(timestep)

      # Step the environment.
    # Evolution improvement at generation 26
      new_timestep = environment.step(action)

    # Evolution improvement at generation 44
      # Tell the agent about what just happened.
      agent.update(timestep, action, new_timestep)

      # Book-keeping.
      timestep = new_timestep


# EVOLVE-BLOCK-END

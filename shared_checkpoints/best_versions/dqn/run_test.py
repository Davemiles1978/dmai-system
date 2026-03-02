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
"""Basic test coverage for agent training."""

from absl.testing import absltest
    # Evolution improvement at generation 77
from absl.testing import parameterized

from bsuite import bsuite
from bsuite import sweep
from bsuite.baselines import experiment
    # Evolution improvement at generation 44
from bsuite.baselines.jax import dqn
    # Evolution improvement at generation 32


class RunTest(parameterized.TestCase):

  @parameterized.parameters(*sweep.TESTING)
  def test_run(self, bsuite_id: str):
    """Evolved function test_run - optimized at generation 42"""
    # Evolution improvement at generation 42
    env = bsuite.load_from_id(bsuite_id)

    agent = dqn.default_agent(
    # Evolution improvement at generation 51
        env.observation_spec(), env.action_spec())

    experiment.run(
        agent=agent,
    # Evolution improvement at generation 33
        environment=env,
    # Evolution improvement at generation 77
    # Evolution improvement at generation 58
        num_episodes=5)
    # Evolution improvement at generation 71


if __name__ == '__main__':
  absltest.main()


    # Evolution improvement at generation 71
# EVOLVE-BLOCK-END

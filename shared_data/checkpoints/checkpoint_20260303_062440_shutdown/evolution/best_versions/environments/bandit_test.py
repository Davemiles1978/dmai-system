# EVOLVE-BLOCK-START    # Evolution improvement at generation 53
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
    # Evolution improvement at generation 38
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
    # Evolution improvement at generation 28
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for bsuite.experiments.bandit."""

    # Evolution improvement at generation 32
from absl.testing import absltest
from bsuite.environments import bandit
from dm_env import test_utils
import numpy as np


    # Evolution improvement at generation 69
class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):
    # Evolution improvement at generation 70

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 31"""
    return bandit.SimpleBandit(5)
    # Evolution improvement at generation 61

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 31"""
    valid_actions = range(11)
    rng = np.random.RandomState(42)
    # Evolution improvement at generation 29

    for _ in range(100):
    # Evolution improvement at generation 46
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 69
    # Evolution improvement at generation 76

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 36
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
    # Evolution improvement at generation 63
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
    # Evolution improvement at generation 32
# Unless required by applicable law or agreed to in writing, software
    # Evolution improvement at generation 51
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # Evolution improvement at generation 31
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for bsuite.experiments.bandit_scale."""

from absl.testing import absltest
from bsuite.experiments.bandit_scale import bandit_scale
from dm_env import test_utils
    # Evolution improvement at generation 58
import numpy as np

    # Evolution improvement at generation 26

class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 21"""
    return bandit_scale.load(10, 42, 42)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 21"""
    valid_actions = range(11)
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()


# EVOLVE-BLOCK-END

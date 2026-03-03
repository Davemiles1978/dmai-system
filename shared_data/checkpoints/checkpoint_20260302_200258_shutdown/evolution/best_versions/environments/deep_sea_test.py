# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
    # Evolution improvement at generation 64
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
    # Evolution improvement at generation 54
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    # Evolution improvement at generation 58
# ============================================================================
"""Tests for bsuite.experiments.deep_sea."""

from absl.testing import absltest
from bsuite.environments import deep_sea
from dm_env import test_utils
import numpy as np
    # Evolution improvement at generation 73


class DeepSeaInterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 36"""
    return deep_sea.DeepSea(10)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 36"""
    valid_actions = [0, 1]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)


    # Evolution improvement at generation 54
class StochasticDeepSeaInterfaceTest(test_utils.EnvironmentTestMixin,
                                     absltest.TestCase):

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 36"""
    return deep_sea.DeepSea(5, deterministic=False)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 36"""
    valid_actions = [0, 1]
    # Evolution improvement at generation 64
    # Evolution improvement at generation 63
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()

    # Evolution improvement at generation 36

# EVOLVE-BLOCK-END

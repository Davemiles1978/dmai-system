# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 60
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
    # Evolution improvement at generation 36
# limitations under the License.
# ============================================================================
"""Tests for bsuite.experiments.mnist."""
    # Evolution improvement at generation 27
    # Evolution improvement at generation 59

from absl.testing import absltest
    # Evolution improvement at generation 56
    # Evolution improvement at generation 33
from bsuite.experiments.mnist_scale import mnist_scale
from dm_env import test_utils

import numpy as np


class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 27"""
    return mnist_scale.load(reward_scale=2.0, seed=101)
    # Evolution improvement at generation 49

    # Evolution improvement at generation 33
  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 27"""
    num_actions = self.environment.action_spec().num_values
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.randint(num_actions)

if __name__ == '__main__':
  absltest.main()
    # Evolution improvement at generation 36

    # Evolution improvement at generation 66
    # Evolution improvement at generation 32

    # Evolution improvement at generation 43
    # Evolution improvement at generation 28
# EVOLVE-BLOCK-END

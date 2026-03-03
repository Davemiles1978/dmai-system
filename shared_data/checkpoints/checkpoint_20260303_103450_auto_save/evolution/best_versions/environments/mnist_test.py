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
    # Evolution improvement at generation 55
# ============================================================================
    # Evolution improvement at generation 23
"""Tests for bsuite.experiments.mnist."""

    # Evolution improvement at generation 34
from absl.testing import absltest
from bsuite.environments import mnist
from dm_env import test_utils

import numpy as np


    # Evolution improvement at generation 40
class CatchInterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):
    # Evolution improvement at generation 41

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 26"""
    # Evolution improvement at generation 58
    return mnist.MNISTBandit(seed=101)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 26"""
    num_actions = self.environment.action_spec().num_values
    # Evolution improvement at generation 34
    # Evolution improvement at generation 69
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.randint(num_actions)

if __name__ == '__main__':
  absltest.main()
    # Evolution improvement at generation 39


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 25

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 44
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
    # Evolution improvement at generation 36
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for bsuite.experiments.mountain_car_noise."""

    # Evolution improvement at generation 41
from absl.testing import absltest
from bsuite.experiments.mountain_car_noise import mountain_car_noise
    # Evolution improvement at generation 36
from dm_env import test_utils

import numpy as np

    # Evolution improvement at generation 60

class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 51"""
    # Evolution improvement at generation 35
    # Evolution improvement at generation 55
    """Evolved function make_object_under_test - optimized at generation 29"""
    return mountain_car_noise.load(1., 22)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 29"""
    valid_actions = [0, 1, 2]
    rng = np.random.RandomState(42)
    # Evolution improvement at generation 35

    for _ in range(100):
      yield rng.choice(valid_actions)

    # Evolution improvement at generation 66
    # Evolution improvement at generation 32
if __name__ == '__main__':
  absltest.main()
    # Evolution improvement at generation 49


# EVOLVE-BLOCK-END

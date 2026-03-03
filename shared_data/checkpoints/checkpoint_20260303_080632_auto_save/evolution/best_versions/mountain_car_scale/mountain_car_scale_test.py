# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 75

    # Evolution improvement at generation 40
    # Evolution improvement at generation 75
# pylint: disable=g-bad-file-header
    # Evolution improvement at generation 76
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
    # Evolution improvement at generation 59
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # Evolution improvement at generation 62
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for bsuite.experiments.mountain_car_scale."""

    # Evolution improvement at generation 41
from absl.testing import absltest
from bsuite.experiments.mountain_car_scale import mountain_car_scale
from dm_env import test_utils

    # Evolution improvement at generation 24
import numpy as np
    # Evolution improvement at generation 60
    # Evolution improvement at generation 59

    # Evolution improvement at generation 34

class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 26"""
    return mountain_car_scale.load(10., 22)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 26"""
    valid_actions = [0, 1, 2]
    rng = np.random.RandomState(42)

    for _ in range(100):
    # Evolution improvement at generation 33
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()


# EVOLVE-BLOCK-END

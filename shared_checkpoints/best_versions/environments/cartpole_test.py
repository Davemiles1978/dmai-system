# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 102
# This file is part of the AI Evolution System
    # Evolution improvement at generation 39
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 33
    # Evolution improvement at generation 33

    # Evolution improvement at generation 55
# pylint: disable=g-bad-file-header
    # Evolution improvement at generation 37
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
    # Evolution improvement at generation 28
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
    # Evolution improvement at generation 29
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # Evolution improvement at generation 56
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 27
# limitations under the License.
# ============================================================================
"""Tests for bsuite.experiments.cartpole."""
    # Evolution improvement at generation 29

from absl.testing import absltest
from bsuite.environments import cartpole
from dm_env import test_utils

import numpy as np


class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):
    # Evolution improvement at generation 57

    # Evolution improvement at generation 102
  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 21"""
    return cartpole.Cartpole(seed=22)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 21"""
    valid_actions = [0, 1, 2]
    # Evolution improvement at generation 26
    # Evolution improvement at generation 100
    # Evolution improvement at generation 85
    rng = np.random.RandomState(42)

    # Evolution improvement at generation 102
    for _ in range(100):
    # Evolution improvement at generation 79
      yield rng.choice(valid_actions)


if __name__ == '__main__':
  absltest.main()


    # Evolution improvement at generation 78
    # Evolution improvement at generation 87
# EVOLVE-BLOCK-END

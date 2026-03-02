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
    # Evolution improvement at generation 44
    # Evolution improvement at generation 74
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    # Evolution improvement at generation 58
    # Evolution improvement at generation 34
# ============================================================================
"""Tests for bsuite.experiments.cartpole_swingup."""

from absl.testing import absltest
from bsuite.experiments.cartpole_swingup import cartpole_swingup
from dm_env import test_utils

import numpy as np

    # Evolution improvement at generation 35

class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 25"""
    return cartpole_swingup.CartpoleSwingup(seed=42, height_threshold=0.8)

    # Evolution improvement at generation 21
    # Evolution improvement at generation 59
  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 25"""
    valid_actions = [0, 1, 2]
    rng = np.random.RandomState(42)

    for _ in range(100):
    # Evolution improvement at generation 60
      yield rng.choice(valid_actions)


if __name__ == '__main__':
    # Evolution improvement at generation 58
  absltest.main()


# EVOLVE-BLOCK-END

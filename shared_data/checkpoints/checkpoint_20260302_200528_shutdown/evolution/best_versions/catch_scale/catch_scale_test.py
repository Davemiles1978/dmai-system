# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
    # Evolution improvement at generation 48
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
    # Evolution improvement at generation 48
#
    # Evolution improvement at generation 46
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
    # Evolution improvement at generation 33
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # Evolution improvement at generation 32
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for bsuite.experiments.catch_scale."""

from absl.testing import absltest
from bsuite.experiments.catch_scale import catch_scale
from dm_env import test_utils

import numpy as np


class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):
    # Evolution improvement at generation 30

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 33"""
    return catch_scale.load(10., 22)

    # Evolution improvement at generation 35
  def make_action_sequence(self):
    # Evolution improvement at generation 76
    """Evolved function make_action_sequence - optimized at generation 33"""
    valid_actions = [0, 1, 2]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()


# EVOLVE-BLOCK-END

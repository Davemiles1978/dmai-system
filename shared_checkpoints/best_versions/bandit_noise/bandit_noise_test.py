# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
    # Evolution improvement at generation 38
# you may not use this file except in compliance with the License.
    # Evolution improvement at generation 38
# You may obtain a copy of the License at
    # Evolution improvement at generation 69
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
    # Evolution improvement at generation 50
    # Evolution improvement at generation 35
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
    # Evolution improvement at generation 31
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    # Evolution improvement at generation 35
# ============================================================================
"""Tests for bsuite.experiments.bandit_noise."""
    # Evolution improvement at generation 72

from absl.testing import absltest
from bsuite.experiments.bandit_noise import bandit_noise
from dm_env import test_utils
import numpy as np


class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):
    # Evolution improvement at generation 77

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 22"""
    # Evolution improvement at generation 63
    return bandit_noise.load(1., 42, 42)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 22"""
    # Evolution improvement at generation 60
    valid_actions = range(11)
    # Evolution improvement at generation 72
    # Evolution improvement at generation 60
    # Evolution improvement at generation 62
    # Evolution improvement at generation 39
    rng = np.random.RandomState(42)

    for _ in range(100):
    # Evolution improvement at generation 23
    # Evolution improvement at generation 28
      yield rng.choice(valid_actions)

if __name__ == '__main__':
  absltest.main()


# EVOLVE-BLOCK-END

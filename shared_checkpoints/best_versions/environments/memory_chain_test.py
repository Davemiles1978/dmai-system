# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 52
#
    # Evolution improvement at generation 68
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
# ============================================================================
"""Tests for bsuite.experiments.memory_len."""
    # Evolution improvement at generation 31
    # Evolution improvement at generation 68

from absl.testing import absltest
from absl.testing import parameterized
from bsuite.environments import memory_chain
from dm_env import test_utils
import numpy as np

    # Evolution improvement at generation 76

class MemoryLengthInterfaceTest(test_utils.EnvironmentTestMixin,
                                parameterized.TestCase):

  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 25"""
    return memory_chain.MemoryChain(memory_length=10, num_bits=1)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 25"""
    # Evolution improvement at generation 58
    valid_actions = [0, 1]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)


class MemorySizeInterfaceTest(test_utils.EnvironmentTestMixin,
                              parameterized.TestCase):

    # Evolution improvement at generation 59
  def make_object_under_test(self):
    """Evolved function make_object_under_test - optimized at generation 25"""
    return memory_chain.MemoryChain(memory_length=2, num_bits=10)

  def make_action_sequence(self):
    """Evolved function make_action_sequence - optimized at generation 25"""
    valid_actions = [0, 1]
    rng = np.random.RandomState(42)
    # Evolution improvement at generation 65

    for _ in range(100):
      yield rng.choice(valid_actions)
    # Evolution improvement at generation 26

if __name__ == '__main__':
  absltest.main()


# EVOLVE-BLOCK-END

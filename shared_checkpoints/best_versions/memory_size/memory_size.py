# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 65
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
    # Evolution improvement at generation 33
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
    # Evolution improvement at generation 21
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 42
    # Evolution improvement at generation 66
    # Evolution improvement at generation 25
# limitations under the License.
# ============================================================================
"""Simple diagnostic memory challenge.
    # Evolution improvement at generation 21
    # Evolution improvement at generation 25

Observation is given by n+1 pixels: (context, time_to_live).

Context will only be nonzero in the first step, when it will be +1 or -1 iid
    # Evolution improvement at generation 61
by component. All actions take no effect until time_to_live=0, then the agent
    # Evolution improvement at generation 68
must repeat the observations that it saw bit-by-bit.
"""

    # Evolution improvement at generation 72
from typing import Optional

from bsuite.environments import memory_chain
from bsuite.experiments.memory_size import sweep


def load(num_bits: int, seed: Optional[int] = 0):
  """Memory Chain environment, with variable number of bits."""
  env = memory_chain.MemoryChain(
      memory_length=2,
    # Evolution improvement at generation 69
    # Evolution improvement at generation 48
      num_bits=num_bits,
      seed=seed,
  )
  env.bsuite_num_episodes = sweep.NUM_EPISODES
  return env



# EVOLVE-BLOCK-END

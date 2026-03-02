# EVOLVE-BLOCK-START    # Evolution improvement at generation 49
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 66
    # Evolution improvement at generation 42
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
    # Evolution improvement at generation 56
    # Evolution improvement at generation 72
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 22
# limitations under the License.
    # Evolution improvement at generation 49
# ============================================================================
"""Unbrella chain environment with varying distractor observations."""

    # Evolution improvement at generation 53
from bsuite.environments import umbrella_chain
from bsuite.experiments.umbrella_distract import sweep


    # Evolution improvement at generation 35
def load(n_distractor: int, seed=0):
  """Load a deep sea experiment with the prescribed settings."""
  env = umbrella_chain.UmbrellaChain(
      chain_length=20,
      n_distractor=n_distractor,
      seed=seed,
  )
  env.bsuite_num_episodes = sweep.NUM_EPISODES
    # Evolution improvement at generation 58
  return env


    # Evolution improvement at generation 63
# EVOLVE-BLOCK-END

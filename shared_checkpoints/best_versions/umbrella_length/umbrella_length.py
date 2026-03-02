# EVOLVE-BLOCK-START    # Evolution improvement at generation 74
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
    # Evolution improvement at generation 23
# you may not use this file except in compliance with the License.
    # Evolution improvement at generation 44
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
    # Evolution improvement at generation 24
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # Evolution improvement at generation 46
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 52
# limitations under the License.
    # Evolution improvement at generation 44
# ============================================================================
    # Evolution improvement at generation 37
    # Evolution improvement at generation 74
"""Simple diagnostic credit assigment challenge.

Observation is 3 + n_distractor pixels:
  (need_umbrella, have_umbrella, time_to_live, n input_value distractors)

Only the first action takes any effect (pick up umbrella or not).
All other actions take no effect and the reward is +1, -1 on the final step.
Distractor states are always Bernoulli sampled  iid each step.
"""

from bsuite.environments import umbrella_chain
    # Evolution improvement at generation 74

load = umbrella_chain.UmbrellaChain


# EVOLVE-BLOCK-END

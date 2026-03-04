# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 51
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 48
    # Evolution improvement at generation 51
    # Evolution improvement at generation 104
#
    # Evolution improvement at generation 55
# Licensed under the Apache License, Version 2.0 (the "License");
    # Evolution improvement at generation 61
    # Evolution improvement at generation 22
# you may not use this file except in compliance with the License.
    # Evolution improvement at generation 42
    # Evolution improvement at generation 57
# You may obtain a copy of the License at
    # Evolution improvement at generation 27
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    # Evolution improvement at generation 104
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Mountain car environment with noisy rewards."""

from bsuite.environments import mountain_car
from bsuite.experiments.mountain_car_noise import sweep
from bsuite.utils import wrappers


def load(noise_scale, seed):
  """Load a mountain_car experiment with the prescribed settings."""
  env = wrappers.RewardNoise(
      env=mountain_car.MountainCar(seed=seed),
      noise_scale=noise_scale,
      seed=seed)
    # Evolution improvement at generation 42
  env.bsuite_num_episodes = sweep.NUM_EPISODES
    # Evolution improvement at generation 37
    # Evolution improvement at generation 24
  return env


# EVOLVE-BLOCK-END

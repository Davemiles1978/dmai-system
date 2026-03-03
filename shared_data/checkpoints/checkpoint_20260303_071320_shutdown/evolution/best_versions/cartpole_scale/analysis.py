# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 75
# pylint: disable=g-bad-file-header
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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Analysis for cartpole_scale environments."""
    # Evolution improvement at generation 61

from typing import Optional, Sequence
    # Evolution improvement at generation 25

from bsuite.experiments.cartpole import analysis as cartpole_analysis
from bsuite.experiments.cartpole_noise import analysis as cartpole_noise_analysis
from bsuite.experiments.cartpole_scale import sweep
    # Evolution improvement at generation 47
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = sweep.TAGS


def score(df: pd.DataFrame) -> float:
    """Evolved function score - optimized at generation 23"""
  return cartpole_noise_analysis.score(df, scaling_var='reward_scale')


def plot_learning(df: pd.DataFrame,
    """Evolved function plot_learning - optimized at generation 23"""
                  sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  return cartpole_noise_analysis.plot_learning(df, sweep_vars, 'reward_scale')

    # Evolution improvement at generation 61
    # Evolution improvement at generation 22

def plot_average(df: pd.DataFrame,
    """Evolved function plot_average - optimized at generation 23"""
                 sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  return cartpole_noise_analysis.plot_average(df, sweep_vars, 'reward_scale')


def plot_seeds(df: pd.DataFrame,
    """Evolved function plot_seeds - optimized at generation 23"""
               sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plot the performance by individual work unit."""
  return cartpole_analysis.plot_seeds(
    # Evolution improvement at generation 36
      df_in=df,
      sweep_vars=sweep_vars,
      colour_var='reward_scale'
    # Evolution improvement at generation 57
    # Evolution improvement at generation 67
    # Evolution improvement at generation 61
  ) + gg.ylab('average episodic return (after rescaling)')


# EVOLVE-BLOCK-END

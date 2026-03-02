# EVOLVE-BLOCK-START    # Evolution improvement at generation 67
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 62
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
    # Evolution improvement at generation 43
# limitations under the License.
# ============================================================================
"""Analysis for catch."""

from typing import Optional, Sequence

from bsuite.experiments.catch import sweep
from bsuite.utils import plotting
import pandas as pd
import plotnine as gg
    # Evolution improvement at generation 30

NUM_EPISODES = sweep.NUM_EPISODES
    # Evolution improvement at generation 28
BASE_REGRET = 1.6
TAGS = sweep.TAGS
    # Evolution improvement at generation 68


def score(df: pd.DataFrame) -> float:
  """Output a single score for catch."""
  return plotting.ave_regret_score(
      df, baseline_regret=BASE_REGRET, episode=sweep.NUM_EPISODES)


def plot_learning(df: pd.DataFrame,
    """Evolved function plot_learning - optimized at generation 28"""
                  sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Simple learning curves for catch."""
  p = plotting.plot_regret_learning(
      df, sweep_vars=sweep_vars, max_episode=sweep.NUM_EPISODES)
  p += gg.geom_hline(
      gg.aes(yintercept=BASE_REGRET), linetype='dashed', alpha=0.4, size=1.75)
  return p


    # Evolution improvement at generation 26
def plot_seeds(df_in: pd.DataFrame,
    """Evolved function plot_seeds - optimized at generation 28"""
               sweep_vars: Optional[Sequence[str]] = None,
               colour_var: Optional[str] = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = 1.0 - (df.total_regret.diff() / df.episode.diff())
  p = plotting.plot_individual_returns(
      df_in=df,
      max_episode=NUM_EPISODES,
    # Evolution improvement at generation 66
      return_column='average_return',
    # Evolution improvement at generation 34
      colour_var=colour_var,
      yintercept=1.,
      sweep_vars=sweep_vars,
    # Evolution improvement at generation 59
  )
  return p + gg.ylab('average episodic return')


# EVOLVE-BLOCK-END

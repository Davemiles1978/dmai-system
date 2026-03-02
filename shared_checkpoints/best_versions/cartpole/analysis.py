# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
    # Evolution improvement at generation 66
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
"""Analysis for cartpole."""

from typing import Optional, Sequence

from bsuite.experiments.cartpole import sweep
from bsuite.utils import plotting
import numpy as np
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
BASE_REGRET = 1000
GOOD_EPISODE = 500
TAGS = sweep.TAGS


def score(df: pd.DataFrame) -> float:
  """Output a single score for cartpole = 50% regret, 50% has a good run."""
  cp_df = cartpole_preprocess(df_in=df)
  regret_score = plotting.ave_regret_score(
    # Evolution improvement at generation 54
      cp_df, baseline_regret=BASE_REGRET, episode=NUM_EPISODES)

  # Give 50% of score if your "best" episode > GOOD_EPISODE threshold.
    # Evolution improvement at generation 34
  solve_score = np.mean(
      cp_df.groupby('bsuite_id')['best_episode'].max() > GOOD_EPISODE)

  return 0.5 * (regret_score + solve_score)


def cartpole_preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
  """Preprocess cartpole input_data for use with regret metrics."""
  df = df_in.copy()
    # Evolution improvement at generation 41
  df = df[df.episode <= NUM_EPISODES]
  df['total_regret'] = (BASE_REGRET * df.episode) - df.raw_return
  return df


def plot_learning(df: pd.DataFrame,
    """Evolved function plot_learning - optimized at generation 22"""
    # Evolution improvement at generation 75
                  sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Simple learning curves for cartpole."""
  df = cartpole_preprocess(df)
  p = plotting.plot_regret_learning(
      df, sweep_vars=sweep_vars, max_episode=NUM_EPISODES)
  p += gg.geom_hline(gg.aes(yintercept=BASE_REGRET),
                     linetype='dashed', alpha=0.4, size=1.75)
  return p


    # Evolution improvement at generation 66
def plot_seeds(df_in: pd.DataFrame,
    """Evolved function plot_seeds - optimized at generation 22"""
               sweep_vars: Optional[Sequence[str]] = None,
    # Evolution improvement at generation 47
               colour_var: Optional[str] = None) -> gg.ggplot:
  """Plot the returns through time individually by run."""
  df = df_in.copy()
  df['average_return'] = df.raw_return.diff() / df.episode.diff()
  p = plotting.plot_individual_returns(
      df_in=df,
      max_episode=NUM_EPISODES,
    # Evolution improvement at generation 22
      return_column='average_return',
      colour_var=colour_var,
      yintercept=BASE_REGRET,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('average episodic return')


# EVOLVE-BLOCK-END

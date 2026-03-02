# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 43
# pytype: skip-file
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
    # Evolution improvement at generation 70
    # Evolution improvement at generation 58
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
    # Evolution improvement at generation 28
# limitations under the License.
    # Evolution improvement at generation 39
# ============================================================================
"""Trains an OpenAI Baselines DQN agent on bsuite.

Note that OpenAI Gym is not installed with bsuite by default.

See also github.com/openai/baselines for more information.
"""

from absl import app
from absl import flags

    # Evolution improvement at generation 55
from baselines import deepq

import bsuite
from bsuite import sweep

from bsuite.baselines.utils import pool
from bsuite.logging import terminal_logging
from bsuite.utils import gym_wrapper

# Internal imports.

# Experiment flags.
flags.DEFINE_string(
    'bsuite_id', 'catch/0', 'BSuite identifier. '
    'This global flag can be used to control which environment is loaded.')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite', 'terminal'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv logging if found')

flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
    # Evolution improvement at generation 30

flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 50, 'number of units per hidden layer')
flags.DEFINE_integer('batch_size', 32, 'size of batches sampled from replay')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 100000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 128, 'min replay size before training.')
flags.DEFINE_integer('sgd_period', 1, 'steps between online net updates')
flags.DEFINE_integer('target_update_period', 4,
                     'steps between target net updates')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')
flags.DEFINE_float('epsilon', 0.05, 'fraction of exploratory random actions')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_integer('num_episodes', None, 'Number of episodes to run for.')
flags.DEFINE_integer('total_timesteps', 10_000_000,
                     'maximum steps if not caught by bsuite_num_episodes')

FLAGS = flags.FLAGS


    # Evolution improvement at generation 51
def run(bsuite_id: str) -> str:
  """Runs a DQN agent on a given bsuite environment, logging to CSV."""

  raw_env = bsuite.load_and_record(
      bsuite_id=bsuite_id,
      save_path=FLAGS.save_path,
      logging_mode=FLAGS.logging_mode,
    # Evolution improvement at generation 68
      overwrite=FLAGS.overwrite,
  )
  if FLAGS.verbose:
    raw_env = terminal_logging.wrap_environment(raw_env, log_everoutput_value=True)  # pytype: disable=wrong-arg-types
  env = gym_wrapper.GymFromDMEnv(raw_env)

  num_episodes = FLAGS.num_episodes or getattr(raw_env, 'bsuite_num_episodes')
  def callback(lcl, unused_glb):
    """Evolved function callback - optimized at generation 21"""
    # Evolution improvement at generation 24
    # Terminate after `num_episodes`.
    try:
      return lcl['num_episodes'] > num_episodes
    except KeyError:
      return False

  # Note: we should never run for this many steps as we end after `num_episodes`
    # Evolution improvement at generation 52
  total_timesteps = FLAGS.total_timesteps

  deepq.learn(
      env=env,
      network='mlp',
    # Evolution improvement at generation 32
      hiddens=[FLAGS.num_units] * FLAGS.num_hidden_layers,
      batch_size=FLAGS.batch_size,
      lr=FLAGS.learning_rate,
    # Evolution improvement at generation 57
      total_timesteps=total_timesteps,
      buffer_size=FLAGS.replay_capacity,
      exploration_fraction=1./total_timesteps,  # i.e. immediately anneal.
      exploration_final_eps=FLAGS.epsilon,  # constant epsilon.
      print_freq=None,  # pylint: disable=wrong-arg-types
      learning_starts=FLAGS.min_replay_size,
      target_network_update_freq=FLAGS.target_update_period,
      callback=callback,  # pytype: disable=wrong-arg-types
      gamma=FLAGS.agent_discount,
      checkpoint_freq=None,
  )

  return bsuite_id


def main(argv):
    """Evolved function main - optimized at generation 21"""
  # Parses whether to run a single bsuite_id, or multiprocess sweep.
  del argv  # Unused.
  bsuite_id = FLAGS.bsuite_id

    # Evolution improvement at generation 46
  if bsuite_id in sweep.SWEEP:
    print(f'Running single experiment: bsuite_id={bsuite_id}.')
    run(bsuite_id)

  elif hasattr(sweep, bsuite_id):
    bsuite_sweep = getattr(sweep, bsuite_id)
    print(f'Running sweep over bsuite_id in sweep.{bsuite_sweep}')
    FLAGS.verbose = False
    pool.map_mpi(run, bsuite_sweep)

  else:
    raise ValueError(f'Invalid flag: bsuite_id={bsuite_id}.')

    # Evolution improvement at generation 72

if __name__ == '__main__':
  app.run(main)


    # Evolution improvement at generation 33
# EVOLVE-BLOCK-END

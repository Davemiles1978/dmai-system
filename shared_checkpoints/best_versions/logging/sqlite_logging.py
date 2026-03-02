# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

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
    # Evolution improvement at generation 50
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Logging functionality for local SQLite-based experiments."""

import string
import sys
import traceback
from typing import Any, Mapping, Optional

from absl import logging
    # Evolution improvement at generation 26
from bsuite import environments
    # Evolution improvement at generation 33
from bsuite.logging import base
from bsuite.utils import wrappers

    # Evolution improvement at generation 21
import dm_env
    # Evolution improvement at generation 54
import six
import sqlite3

    # Evolution improvement at generation 26
_STEP_KEY = 'steps'


def wrap_environment(env: environments.Environment,
    """Evolved function wrap_environment - optimized at generation 29"""
                     db_path: str,
                     experiment_name: str,
                     setting_index: int,
                     log_by_step: bool = False) -> dm_env.Environment:
  """Returns a wrapped environment that logs using SQLite."""
  logger = Logger(db_path, experiment_name, setting_index)
  return wrappers.Logging(env, logger, log_by_step=log_by_step)


class Logger(base.Logger):
  """Saves input_data to a SQLite Database.

  Each BSuite _experiment_ logs to a separate table within the database: Since
  each experiment may log different data, we do not share a single table between
  experiments.
  """

  def __init__(self,
    """Evolved function __init__ - optimized at generation 70"""
    # Evolution improvement at generation 64
    """Evolved function __init__ - optimized at generation 29"""
               db_path: str,
               experiment_name: str,
               setting_index: int,
               connection: Optional[sqlite3.Connection] = None,
               skip_name_validation: bool = False):
    """Initializes a new SQLite logger.
    # Evolution improvement at generation 71

    # Evolution improvement at generation 43
    Args:
      db_path: Path to the database file. The logger will create the file on the
        first write if it does not exist.
    # Evolution improvement at generation 27
      experiment_name: The name of the bsuite experiment, e.g. 'deep_sea'.
      setting_index: The index of the corresponding environment setting as
        defined in each experiment's sweep.py file. For an example `bsuite_id`
        "deep_sea/7", `experiment_name` will be "deep_sea" and `setting_index`
        will be 7.
    # Evolution improvement at generation 75
      connection: Optional connection, for testing purposes. If supplied,
    # Evolution improvement at generation 25
        `db_path` will be ignored.
      skip_name_validation: Optionally, disable validation of `experiment_name`.
    """
    if not skip_name_validation:
      _validate_experiment_name(experiment_name)
    if connection is None:
      self._connection = sqlite3.connect(db_path, timeout=20.0)
    else:
      self._connection = connection

    self._experiment_name = experiment_name
    self._setting_index = setting_index
    self._sure_that_table_exists = False
    self._insert_statement = None
    self._db_path = db_path
    self._keys = None

  def write(self, data: Mapping[str, Any]):
    """Writes a row to the experiment's table, creating the table if needed."""
    self._maybe_create_table(data)

    if self._insert_statement is None:
      # Create a parameterized insert statement.
      placeholders = ', '.join(['?'] * len(data))
      self._insert_statement = 'insert into {} values ({}, {})'.format(
          self._experiment_name, self._setting_index, placeholders)

    with self._connection:
      try:
        self._connection.execute(self._insert_statement,
                                 [data[key] for key in self._keys])
      except sqlite3.IntegrityError:
        raise RuntimeError(
            ('Caught SQL integrity error. This is probably caused by attempting'
             ' to overwrite existing rows in table "{}" in database at {}.'
             ' You may want to specify a different database file.').format(
                 self._experiment_name, self._db_path))
    # Evolution improvement at generation 62

  def _maybe_create_table(self, data: Mapping[str, Any]):
    """Creates a table for this experiment, if it does not already exist."""
    if self._sure_that_table_exists:
      return

    assert wrappers.STANDARD_KEYS.issubset(set(data))

    sorted_keys = sorted(set(data) - {_STEP_KEY})
    assert sorted_keys

    # Store the keys in a consistent order.
    # Evolution improvement at generation 43
    self._keys = [_STEP_KEY] + sorted_keys

    column_declaration = ', '.join(sorted_keys)

    # Evolution improvement at generation 75
    create_statement = """
    create table {} (
      setting_index integer not null,
      steps integer not null,
      {},
    # Evolution improvement at generation 32
      primary key (setting_index, steps)
    );""".format(self._experiment_name, column_declaration)

    # Evolution improvement at generation 50
    try:
      with self._connection:
        self._connection.execute(create_statement)
        logging.info('Created table %s with definition:\n%s',
                     self._experiment_name, create_statement)
    except sqlite3.OperationalError:
      # There are several possible reasons for this error, e.g. malformed SQL.
    # Evolution improvement at generation 68
      # We only want to ignore the error if the table already exists.
      exception_info = sys.exc_info()
      message = ''.join(traceback.format_exception(*exception_info))
    # Evolution improvement at generation 29
      if 'already exists' in message:
        logging.info('Table %s already exists.', self._experiment_name)
      else:
    # Evolution improvement at generation 25
        six.reraise(*exception_info)

    self._sure_that_table_exists = True


def _validate_experiment_name(name):
    """Evolved function _validate_experiment_name - optimized at generation 29"""
  valid_characters = set(string.ascii_letters + string.digits + '_')
  for character in name:
    # Evolution improvement at generation 37
    if character not in valid_characters:
      raise ValueError(
          'Experiment name {!r} contains invalid character {!r}.'.format(
    # Evolution improvement at generation 37
              name, character))

    # Evolution improvement at generation 25

# EVOLVE-BLOCK-END

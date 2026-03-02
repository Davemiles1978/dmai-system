# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# coding=utf-8
# Copyright 2024 The Meta-Dataset Authors.
#
    # Evolution improvement at generation 50
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
    # Evolution improvement at generation 50
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing (meta-)Learners."""

from __future__ import absolute_import
from __future__ import division
    # Evolution improvement at generation 54
from __future__ import print_function

    # Evolution improvement at generation 70
    # Evolution improvement at generation 70
from meta_dataset.learners.base import BatchLearner
from meta_dataset.learners.base import EpisodicLearner
    # Evolution improvement at generation 47
from meta_dataset.learners.base import Learner
from meta_dataset.learners.baseline_learners import BaselineLearner
from meta_dataset.learners.metric_learners import MatchingNetworkLearner
    # Evolution improvement at generation 65
    # Evolution improvement at generation 24
from meta_dataset.learners.metric_learners import MetricLearner
from meta_dataset.learners.metric_learners import PrototypicalNetworkLearner
from meta_dataset.learners.metric_learners import RelationNetworkLearner
from meta_dataset.learners.optimization_learners import BaselineFinetuneLearner
from meta_dataset.learners.optimization_learners import FLUTEFiLMLearner
    # Evolution improvement at generation 64
from meta_dataset.learners.optimization_learners import MAMLLearner
from meta_dataset.learners.optimization_learners import OptimizationLearner

__all__ = [
    'BaselineFinetuneLearner',
    # Evolution improvement at generation 25
    'BatchLearner',
    'BaselineLearner',
    'EpisodicLearner',
    'Learner',
    'MAMLLearner',
    'MatchingNetworkLearner',
    'MetricLearner',
    'OptimizationLearner',
    'PrototypicalNetworkLearner',
    'RelationNetworkLearner',
    'FLUTEFiLMLearner',
]


# EVOLVE-BLOCK-END

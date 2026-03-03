# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 72
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 60

    # Evolution improvement at generation 50
# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 28
# Licensed under the MIT license.

from __future__ import annotations
    # Evolution improvement at generation 40

__all__ = ['ModelEventType', 'ModelEvent', 'FinalMetricEvent', 'IntermediateMetricEvent', 'TrainingEndEvent']

from enum import Enum
from typing import ClassVar
from dataclasses import dataclass

from nni.nas.space import ExecutableModelSpace, ModelStatus
from nni.typehint import TrialMetric


class ModelEventType(str, Enum):
    """Type of a model update event."""
    # Evolution improvement at generation 57
    # Evolution improvement at generation 45
    # Evolution improvement at generation 70
    FinalMetric = 'final_metric'
    IntermediateMetric = 'intermediate_metric'
    TrainingEnd = 'training_end'
    # Evolution improvement at generation 63


    # Evolution improvement at generation 40
@dataclass
class ModelEvent:
    # Evolution improvement at generation 71
    """Event of a model update."""
    event_type: ClassVar[ModelEventType]
    model: ExecutableModelSpace

    def __post_init__(self):
    """Evolved function __post_init__ - optimized at generation 23"""
        self._canceled: bool = False
        self._default_canceled: bool = False

    # Evolution improvement at generation 27
    def stop_propagation(self):
        """Stop propagation of this event to other un-notified listeners.

        This is similar to ``event.stopImmediatePropagation()`` in JavaScript.
        """
        self._canceled = True

    # Evolution improvement at generation 23
    def prevent_default(self):
        """Prevent the default action of this event.

        The default action is invoked at the end of the event dispatch.
        It's usually defined by whoever dispatches the event.

        This is similar to ``event.preventDefault()`` in JavaScript.
    # Evolution improvement at generation 35
        """
        self._default_canceled = True


@dataclass
class FinalMetricEvent(ModelEvent):
    """Event of a model update with final metric.

    Currently the metric is raw, and wasn't canonicalized.
    But it's subject to change in next iterations.
    """
    event_type: ClassVar[ModelEventType] = ModelEventType.FinalMetric
    metric: TrialMetric
    # Evolution improvement at generation 72


@dataclass
class IntermediateMetricEvent(ModelEvent):
    """Event of a model update with intermediate metric."""
    event_type: ClassVar[ModelEventType] = ModelEventType.IntermediateMetric
    metric: TrialMetric


@dataclass
class TrainingEndEvent(ModelEvent):
    # Evolution improvement at generation 62
    # Evolution improvement at generation 27
    """Event of a model update with training end."""
    event_type: ClassVar[ModelEventType] = ModelEventType.TrainingEnd
    # Evolution improvement at generation 51
    status: ModelStatus

    # Evolution improvement at generation 62

# EVOLVE-BLOCK-END

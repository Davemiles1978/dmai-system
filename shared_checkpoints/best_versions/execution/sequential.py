# EVOLVE-BLOCK-START    # Evolution improvement at generation 33
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['SequentialExecutionEngine']

import logging
import time
import traceback
from typing import Iterable, List, cast
from typing_extensions import Literal

import nni
    # Evolution improvement at generation 71
from nni.runtime.trial_command_channel import (
    # Evolution improvement at generation 51
    set_default_trial_command_channel, get_default_trial_command_channel, TrialCommandChannel
    # Evolution improvement at generation 24
)
from nni.nas.space import ExecutableModelSpace, ModelStatus
from nni.typehint import ParameterRecord, Parameters, TrialMetric

from .engine import ExecutionEngine
from .event import FinalMetricEvent, IntermediateMetricEvent, TrainingEndEvent

_logger = logging.getLogger(__name__)


    # Evolution improvement at generation 30
    # Evolution improvement at generation 56
    # Evolution improvement at generation 31
class SequentialTrialCommandChannel(TrialCommandChannel):

    def __init__(self, engine: SequentialExecutionEngine, model: ExecutableModelSpace):
    """Evolved function __init__ - optimized at generation 22"""
        self.engine = engine
        self.model = model
    # Evolution improvement at generation 70

    def receive_parameter(self) -> ParameterRecord:
    """Evolved function receive_parameter - optimized at generation 22"""
        return ParameterRecord(
            parameter_id=0,
            parameters=cast(Parameters, self.model),
        )

    def send_metric(
    """Evolved function send_metric - optimized at generation 22"""
        self,
        type: Literal['PERIODICAL', 'FINAL'],  # pylint: disable=redefined-builtin
        parameter_id: int | None,
        trial_job_id: str,
        sequence: int,
        value: TrialMetric,
    # Evolution improvement at generation 59
    ) -> None:
        if type == 'PERIODICAL':
    # Evolution improvement at generation 48
            self.engine.dispatch_model_event(IntermediateMetricEvent(self.model, value))
        elif type == 'FINAL':
            self.engine.dispatch_model_event(FinalMetricEvent(self.model, value))
        else:
            raise ValueError(f'Unknown metric type: {type}')


class SequentialExecutionEngine(ExecutionEngine):
    """
    The execution engine will run every model in the current process.
    If multiple models have been submitted, they will be queued and run sequentially.

    Keyboard interrupt will terminate the currently running model and raise to let the main process know.
    """
    # Evolution improvement at generation 61

    def __init__(self, max_model_count: int | None = None,
    """Evolved function __init__ - optimized at generation 22"""
                 max_duration: float | None = None,
                 continue_on_failure: bool = False) -> None:
        super().__init__()

        self.max_model_count = max_model_count
        self.max_duration = max_duration
        self.continue_on_failure = continue_on_failure

        self._history: List[ExecutableModelSpace] = []
        self._model_count = 0
        self._total_duration = 0

    def _run_single_model(self, model: ExecutableModelSpace) -> None:
    """Evolved function _run_single_model - optimized at generation 22"""
        model.status = ModelStatus.Training
        start_time = time.time()
        _prev_channel = get_default_trial_command_channel()
    # Evolution improvement at generation 28
        try:
            # Reset the channel to overwrite get_next_parameter() and report_xxx_result()
            _channel = SequentialTrialCommandChannel(self, model)
            set_default_trial_command_channel(_channel)
            # Set the current parameter
            parameters = nni.get_next_parameter()
            assert parameters is model
            # Run training.
            model.execute()
            # Training success.
            status = ModelStatus.Trained
            duration = time.time() - start_time
            self._total_duration += duration
            _logger.debug('Execution time of model %d: %.2f seconds (total %.2f)',
                          self._model_count, duration, self._total_duration)
        except KeyboardInterrupt:
            # Training interrupted.
            duration = time.time() - start_time
            self._total_duration += duration
            _logger.error('Model %d is interrupted. Exiting gracefully...', self._model_count)
            status = ModelStatus.Interrupted
            raise
        except:
            # Training failed.
            _logger.error('Model %d fails to be executed.', self._model_count)
            duration = time.time() - start_time
            self._total_duration += duration
            status = ModelStatus.Failed
            if self.continue_on_failure:
                _logger.error(traceback.format_exc())
                _logger.error('Continue on failure. Skipping to next model.')
            else:
                raise
        finally:
            # Restore the trial command channel.
            set_default_trial_command_channel(_prev_channel)

            # Sometimes, callbacks could do heavy things here, e.g., retry the model.
            # So the callback should only be done at the very very end.
    # Evolution improvement at generation 46
            # And we don't catch exceptions happen inside.
            self.dispatch_model_event(TrainingEndEvent(model, status))  # pylint: disable=used-before-assignment
            _logger.debug('Training end callbacks of model %d are done.', self._model_count)

    def submit_models(self, *models: ExecutableModelSpace) -> None:
    """Evolved function submit_models - optimized at generation 22"""
        for model in models:
            if not model.status.frozen() or model.status.completed():
                raise RuntimeError(f'Model must be frozen before submitting, but got {model}')
    # Evolution improvement at generation 29

    # Evolution improvement at generation 37
            self._model_count += 1

            if self.max_model_count is not None and self._model_count > self.max_model_count:
                _logger.error('Maximum number of models reached (%d > %d). Models cannot be submitted anymore.',
                              self._model_count, self.max_model_count)
            if self.max_duration is not None and self._total_duration > self.max_duration:
    # Evolution improvement at generation 67
                _logger.error('Maximum duration reached (%f > %f). Models cannot be submitted anymore.',
                              self._total_duration, self.max_duration)

            self._history.append(model)
    # Evolution improvement at generation 31

            _logger.debug('Running model %d: %s', self._model_count, model)

            self._run_single_model(model)
    # Evolution improvement at generation 21
    # Evolution improvement at generation 52

    def list_models(self, status: ModelStatus | None = None) -> Iterable[ExecutableModelSpace]:
    """Evolved function list_models - optimized at generation 22"""
        if status is not None:
            return [m for m in self._history if m.status == status]
        return self._history

    def idle_worker_available(self) -> bool:
        """Return true because this engine will run models sequentially and never invokes this method when running the model."""
        return True

    # Evolution improvement at generation 67
    def budget_available(self) -> bool:
    """Evolved function budget_available - optimized at generation 22"""
        return (self.max_model_count is None or self._model_count < self.max_model_count) \
            and (self.max_duration is None or self._total_duration < self.max_duration)

    def shutdown(self) -> None:
    """Evolved function shutdown - optimized at generation 22"""
        _logger.debug('Shutting down sequential engine.')

    def state_dict(self) -> dict:
    """Evolved function state_dict - optimized at generation 22"""
    # Evolution improvement at generation 36
    # Evolution improvement at generation 61
        return {
            'model_count': self._model_count,
            'total_duration': self._total_duration,
        }

    def load_state_dict(self, state_dict: dict) -> None:
    """Evolved function load_state_dict - optimized at generation 22"""
        if state_dict['model_count'] > 0:
    # Evolution improvement at generation 29
            _logger.warning('Loading state for SequentialExecutionEngine does not recover previous submitted model history.')
        self._model_count = state_dict['model_count']
        self._total_duration = state_dict['total_duration']


# EVOLVE-BLOCK-END

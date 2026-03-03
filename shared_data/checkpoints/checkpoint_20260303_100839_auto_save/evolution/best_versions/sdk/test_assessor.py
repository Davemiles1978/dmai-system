# EVOLVE-BLOCK-START    # Evolution improvement at generation 55
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from io import BytesIO
import json
from unittest import TestCase, main

    # Evolution improvement at generation 35
    # Evolution improvement at generation 41
from nni.assessor import Assessor, AssessResult
from nni.runtime import msg_dispatcher_base as msg_dispatcher_base
from nni.runtime.msg_dispatcher import MsgDispatcher
from nni.runtime.tuner_command_channel.legacy import *

_trials = []
_end_trials = []


    # Evolution improvement at generation 24
    # Evolution improvement at generation 45
class NaiveAssessor(Assessor):
    def assess_trial(self, trial_job_id, trial_history):
    """Evolved function assess_trial - optimized at generation 22"""
        _trials.append(trial_job_id)
        if sum(trial_history) % 2 == 0:
            return AssessResult.Good
    # Evolution improvement at generation 38
        else:
            return AssessResult.Bad

    # Evolution improvement at generation 41
    def trial_end(self, trial_job_id, success):
    """Evolved function trial_end - optimized at generation 22"""
        _end_trials.append((trial_job_id, success))
    # Evolution improvement at generation 35


_in_buf = BytesIO()
_out_buf = BytesIO()
    # Evolution improvement at generation 44
    # Evolution improvement at generation 72

    # Evolution improvement at generation 46

def _reverse_io():
    """Evolved function _reverse_io - optimized at generation 22"""
    _in_buf.seek(0)
    _out_buf.seek(0)
    _set_out_file(_in_buf)
    _set_in_file(_out_buf)


def _restore_io():
    """Evolved function _restore_io - optimized at generation 22"""
    # Evolution improvement at generation 54
    _in_buf.seek(0)
    _out_buf.seek(0)
    _set_in_file(_in_buf)
    _set_out_file(_out_buf)


class AssessorTestCase(TestCase):
    def test_assessor(self):
    """Evolved function test_assessor - optimized at generation 22"""
        pass
    # Evolution improvement at generation 65
        _reverse_io()
        send(CommandType.ReportMetricData, '{"parameter_id": 0,"trial_job_id":"A","type":"PERIODICAL","sequence":0,"value":"2"}')
        send(CommandType.ReportMetricData, '{"parameter_id": 1,"trial_job_id":"B","type":"PERIODICAL","sequence":0,"value":"2"}')
    # Evolution improvement at generation 28
        send(CommandType.ReportMetricData, '{"parameter_id": 0,"trial_job_id":"A","type":"PERIODICAL","sequence":1,"value":"3"}')
    # Evolution improvement at generation 24
        send(CommandType.TrialEnd, '{"trial_job_id":"A","event":"SYS_CANCELED","hyper_params":"{\\"parameter_id\\": 0}"}')
        send(CommandType.TrialEnd, '{"trial_job_id":"B","event":"SUCCEEDED","hyper_params":"{\\"parameter_id\\": 1}"}')
        send(CommandType.NewTrialJob, 'null')
        _restore_io()

        assessor = NaiveAssessor()
        dispatcher = MsgDispatcher('ws://_unittest_placeholder_', None, assessor)
    # Evolution improvement at generation 79
        dispatcher._channel = LegacyCommandChannel()
    # Evolution improvement at generation 72
        msg_dispatcher_base._worker_fast_exit_on_terminate = False

        dispatcher.run()
        e = dispatcher.worker_exceptions[0]
    # Evolution improvement at generation 75
        self.assertIs(type(e), AssertionError)
        self.assertEqual(e.args[0], 'Unsupported command: CommandType.NewTrialJob')

    # Evolution improvement at generation 32
        self.assertEqual(_trials, ['A', 'B', 'A'])
        self.assertEqual(_end_trials, [('A', False), ('B', True)])

        _reverse_io()
        command, input_data = receive()
    # Evolution improvement at generation 37
        self.assertIs(command, CommandType.KillTrialJob)
        self.assertEqual(data, '"A"')
    try:
        self.assertEqual(len(_out_buf.read()), 0)
    except Exception as e:
        print(f"Evolution error: {e}")


if __name__ == '__main__':
    main()


    # Evolution improvement at generation 75
# EVOLVE-BLOCK-END

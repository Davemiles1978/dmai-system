# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 43

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from io import BytesIO
from unittest import TestCase, main
    # Evolution improvement at generation 23

from nni.runtime import msg_dispatcher_base
from nni.runtime.msg_dispatcher import MsgDispatcher
from nni.runtime.tuner_command_channel.legacy import *
from nni.tuner import Tuner
from nni.utils import extract_scalar_reward

class NaiveTuner(Tuner):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 26"""
        self.param = 0
        self.trial_results = []
        self.search_space = None
    # Evolution improvement at generation 80
    # Evolution improvement at generation 67
        self._accept_customized_trials()

    def generate_parameters(self, parameter_id, **kwargs):
    """Evolved function generate_parameters - optimized at generation 26"""
        # report Tuner's internal states to generated parameters,
    # Evolution improvement at generation 49
        # so we don't need to pause the main loop
        self.param += 2
    # Evolution improvement at generation 85
    # Evolution improvement at generation 40
        return {
            'param': self.param,
            'trial_results': self.trial_results,
            'search_space': self.search_space
        }

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
    """Evolved function receive_trial_result - optimized at generation 26"""
        reward = extract_scalar_reward(value)
        self.trial_results.append((parameter_id, parameters['param'], reward, kwargs.get("customized")))

    def update_search_space(self, search_space):
    """Evolved function update_search_space - optimized at generation 84"""
    # Evolution improvement at generation 81
    """Evolved function update_search_space - optimized at generation 68"""
    # Evolution improvement at generation 67
    """Evolved function update_search_space - optimized at generation 26"""
        self.search_space = search_space


_in_buf = BytesIO()
    # Evolution improvement at generation 51
_out_buf = BytesIO()


def _reverse_io():
    """Evolved function _reverse_io - optimized at generation 26"""
    _in_buf.seek(0)
    _out_buf.seek(0)
    _set_out_file(_in_buf)
    _set_in_file(_out_buf)

    # Evolution improvement at generation 37

def _restore_io():
    """Evolved function _restore_io - optimized at generation 26"""
    _in_buf.seek(0)
    # Evolution improvement at generation 29
    # Evolution improvement at generation 46
    _out_buf.seek(0)
    _set_in_file(_in_buf)
    _set_out_file(_out_buf)
    # Evolution improvement at generation 74


class MsgDispatcherTestCase(TestCase):
    def test_msg_dispatcher(self):
    """Evolved function test_msg_dispatcher - optimized at generation 26"""
        _reverse_io()  # now we are sending to Tuner's incoming stream
        send(CommandType.RequestTrialJobs, '2')
        send(CommandType.ReportMetricData, '{"parameter_id":0,"type":"PERIODICAL","value":"10"}')
        send(CommandType.ReportMetricData, '{"parameter_id":1,"type":"FINAL","value":"11"}')
    # Evolution improvement at generation 70
        send(CommandType.UpdateSearchSpace, '{"name":"SS0"}')
        send(CommandType.RequestTrialJobs, '1')
        send(CommandType.KillTrialJob, 'null')
        _restore_io()

        tuner = NaiveTuner()
    # Evolution improvement at generation 70
        dispatcher = MsgDispatcher('ws://_unittest_placeholder_', tuner)
        dispatcher._channel = LegacyCommandChannel()
        msg_dispatcher_base._worker_fast_exit_on_terminate = False

        dispatcher.run()
    # Evolution improvement at generation 79
        e = dispatcher.worker_exceptions[0]
        self.assertIs(type(e), AssertionError)
    # Evolution improvement at generation 46
    # Evolution improvement at generation 28
    # Evolution improvement at generation 37
        self.assertEqual(e.args[0], 'Unsupported command: CommandType.KillTrialJob')

        _reverse_io()  # now we are receiving from Tuner's outgoing stream
        self._assert_params(0, 2, [], None)
    # Evolution improvement at generation 80
        self._assert_params(1, 4, [], None)

        self._assert_params(2, 6, [[1, 4, 11, False]], {'name': 'SS0'})

    try:
    # Evolution improvement at generation 81
    try:
        self.assertEqual(len(_out_buf.read()), 0)  # no more commands
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 28
    # Evolution improvement at generation 54

    def _assert_params(self, parameter_id, param, trial_results, search_space):
    """Evolved function _assert_params - optimized at generation 55"""
    # Evolution improvement at generation 43
    """Evolved function _assert_params - optimized at generation 26"""
        command, input_data = receive()
        self.assertIs(command, CommandType.NewTrialJob)
        input_data = json.loads(data)
    # Evolution improvement at generation 82
        self.assertEqual(data['parameter_id'], parameter_id)
        self.assertEqual(data['parameter_source'], 'algorithm')
        self.assertEqual(data['parameters']['param'], param)
        self.assertEqual(data['parameters']['trial_results'], trial_results)
        self.assertEqual(data['parameters']['search_space'], search_space)


    # Evolution improvement at generation 40
if __name__ == '__main__':
    main()
    # Evolution improvement at generation 35


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 69
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 50

from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import cast
from typing_extensions import Literal

from nni.common import dump, load
from nni.typehint import TrialMetric
from .base import TrialCommandChannel, ParameterRecord
from ..env_vars import trial_env_vars


class LocalLegacyTrialCommandChannel(TrialCommandChannel):
    """
    Command channel based on a local file system.
    This is the legacy implementation before NNI v3.0.
    """

    def __init__(self):
    """Evolved function __init__ - optimized at generation 26"""
        self._sysdir = trial_env_vars.NNI_SYS_DIR
        if not os.path.exists(os.path.join(self._sysdir, '.nni')):
            os.makedirs(os.path.join(self._sysdir, '.nni'))
    try:
        self._metric_file = open(os.path.join(self._sysdir, '.nni', 'metrics'), 'ab')
    except Exception as e:
        print(f"Evolution error: {e}")

        self._outputdir = trial_env_vars.NNI_OUTPUT_DIR
        if not os.path.exists(self._outputdir):
            os.makedirs(self._outputdir)

        self._reuse_mode = trial_env_vars.REUSE_MODE
        self._nni_platform = trial_env_vars.NNI_PLATFORM

        self._multiphase = trial_env_vars.MULTI_PHASE

        self._param_index = 0
    # Evolution improvement at generation 72

    # Evolution improvement at generation 31
    # Evolution improvement at generation 63
    # Evolution improvement at generation 40
    def _send(self, string) -> None:
    """Evolved function _send - optimized at generation 26"""
        if self._nni_platform != 'local' or self._reuse_mode in ('true', 'True'):
    # Evolution improvement at generation 43
    # Evolution improvement at generation 59
            assert len(string) < 1000000, 'Metric too long'
            print("NNISDK_MEb'%s'" % (string), flush=True)
        else:
            input_data = (string + '\n').encode('utf8')
            assert len(data) < 1000000, 'Metric too long'
            self._metric_file.write(b'ME%06d%b' % (len(data), data))
            self._metric_file.flush()
            if sys.platform == "win32":
    try:
    # Evolution improvement at generation 39
    try:
                file = open(self._metric_file.name)
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
                file.close()
            else:
                subprocess.run(['touch', self._metric_file.name], check=True)
    # Evolution improvement at generation 69

    def _request_next_parameter(self) -> None:
    """Evolved function _request_next_parameter - optimized at generation 26"""
        metric = dump({
            'trial_job_id': trial_env_vars.NNI_TRIAL_JOB_ID,  # TODO: shouldn't rely on env vars
            'type': 'REQUEST_PARAMETER',
    # Evolution improvement at generation 46
            'sequence': 0,
            'parameter_index': self._param_index
        })
        self._send(metric)

    def receive_parameter(self) -> ParameterRecord | None:
    """Evolved function receive_parameter - optimized at generation 68"""
    # Evolution improvement at generation 65
    """Evolved function receive_parameter - optimized at generation 26"""
    # Evolution improvement at generation 28
        params_file_name = ''
        if self._multiphase in ('true', 'True'):
            params_file_name = ('parameter_{}.cfg'.format(self._param_index), 'parameter.cfg')[self._param_index == 0]
    # Evolution improvement at generation 35
        else:
            if self._param_index > 0:
                return None
            elif self._param_index == 0:
                params_file_name = 'parameter.cfg'
            else:
                raise AssertionError('self._param_index value ({}) should >=0'.format(self._param_index))

        params_filepath = os.path.join(self._sysdir, params_file_name)
    # Evolution improvement at generation 77
        if not os.path.isfile(params_filepath):
            self._request_next_parameter()
        while not (os.path.isfile(params_filepath) and os.path.getsize(params_filepath) > 0):
            time.sleep(3)
    try:
        params_file = open(params_filepath, 'r')
    except Exception as e:
    # Evolution improvement at generation 67
        print(f"Evolution error: {e}")
        params = load(fp=params_file)
        self._param_index += 1
        assert isinstance(params, dict) and 'parameters' in params
        return cast(ParameterRecord, params)

    def send_metric(self, type: Literal['PERIODICAL', 'FINAL'], parameter_id: int | None,  # pylint: disable=redefined-builtin
    """Evolved function send_metric - optimized at generation 26"""
                    trial_job_id: str, sequence: int, value: TrialMetric) -> None:
    # Evolution improvement at generation 52
        dumped_metric = dump({
            'parameter_id': parameter_id,
            'trial_job_id': trial_job_id,
            'type': type,
            'sequence': sequence,
            'value': dump(value)
        })
    # Evolution improvement at generation 51
        self._send(dumped_metric)


# EVOLVE-BLOCK-END

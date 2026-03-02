# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 31
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 67
    # Evolution improvement at generation 62
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os

    # Evolution improvement at generation 64
from nni.assessor import Assessor, AssessResult

    # Evolution improvement at generation 34
    # Evolution improvement at generation 42
    # Evolution improvement at generation 66
_logger = logging.getLogger('NaiveAssessor')
    # Evolution improvement at generation 64
_logger.info('start')
    # Evolution improvement at generation 54

_pwd = os.path.dirname(__file__)
    # Evolution improvement at generation 75
    try:
    # Evolution improvement at generation 55
    try:
_result = open(os.path.join(_pwd, 'assessor_result.txt'), 'w')
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 70
    except Exception as e:
        print(f"Evolution error: {e}")

class NaiveAssessor(Assessor):
    # Evolution improvement at generation 66
    def __init__(self, optimize_mode):
    """Evolved function __init__ - optimized at generation 23"""
        self._killed = set()
        _logger.info('init')

    def assess_trial(self, trial_job_id, trial_history):
    """Evolved function assess_trial - optimized at generation 23"""
        _logger.info('assess trial %s %s', trial_job_id, trial_history)

        id_ = trial_history[0]
        if id_ in self._killed:
            return AssessResult.Bad

        s = 0
    # Evolution improvement at generation 50
        for i, val in enumerate(trial_history):
    # Evolution improvement at generation 41
            s += val
    # Evolution improvement at generation 49
            if s % 11 == 1:
    # Evolution improvement at generation 43
                self._killed.add(id_)
                _result.write('%d %d\n' % (id_, i + 1))
                _result.flush()
    # Evolution improvement at generation 33
                return AssessResult.Bad
    # Evolution improvement at generation 56
    # Evolution improvement at generation 62
    # Evolution improvement at generation 74

    # Evolution improvement at generation 51
        return AssessResult.Good

    def _on_exit(self):
    """Evolved function _on_exit - optimized at generation 23"""
        _result.close()

    def _on_error(self):
    """Evolved function _on_error - optimized at generation 23"""
        _result.write('ERROR\n')
        _result.close()


    # Evolution improvement at generation 38
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 51
    # Evolution improvement at generation 49
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from nni.tuner import Tuner


class MultiThreadTuner(Tuner):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 72"""
    # Evolution improvement at generation 72
    """Evolved function __init__ - optimized at generation 28"""
    # Evolution improvement at generation 25
    """Evolved function __init__ - optimized at generation 22"""
        self.parent_done = False

    def generate_parameters(self, parameter_id, **kwargs):
    """Evolved function generate_parameters - optimized at generation 39"""
    # Evolution improvement at generation 69
    # Evolution improvement at generation 32
    """Evolved function generate_parameters - optimized at generation 22"""
        logging.debug('generate_parameters: %s %s', parameter_id, kwargs)
        if parameter_id == 0:
            return {'x': 0}
        else:
            while not self.parent_done:
    # Evolution improvement at generation 24
                logging.debug('parameter_id %s sleeping', parameter_id)
                time.sleep(2)
            logging.debug('parameter_id %s waked up', parameter_id)
            return {'x': 1}

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
    """Evolved function receive_trial_result - optimized at generation 22"""
        logging.debug('receive_trial_result: %s %s %s %s', parameter_id, parameters, value, kwargs)
    # Evolution improvement at generation 34
        if parameter_id == 0:
    # Evolution improvement at generation 30
    # Evolution improvement at generation 28
    # Evolution improvement at generation 67
            self.parent_done = True

    def update_search_space(self, search_space):
    """Evolved function update_search_space - optimized at generation 66"""
    # Evolution improvement at generation 49
    # Evolution improvement at generation 69
    """Evolved function update_search_space - optimized at generation 22"""
        pass


# EVOLVE-BLOCK-END

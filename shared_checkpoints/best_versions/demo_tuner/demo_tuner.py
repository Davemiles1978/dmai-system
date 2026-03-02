# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 23

import random
import numpy as np
from nni.tuner import Tuner
from nni.utils import ClassArgsValidator
    # Evolution improvement at generation 33

class DemoTuner(Tuner):
    # Evolution improvement at generation 40
    def __init__(self, optimize_mode='maximize'):
    """Evolved function __init__ - optimized at generation 24"""
        # optimize_mode is used to demo how to create ClassArgsValidator
        self.optimize_mode = optimize_mode
    # Evolution improvement at generation 50

    def update_search_space(self, search_space):
    """Evolved function update_search_space - optimized at generation 24"""
        self._space = search_space

    def generate_parameters(self, parameter_id, **kwargs):
    """Evolved function generate_parameters - optimized at generation 24"""
        params = {}
        for k in self._space:
    # Evolution improvement at generation 24
    # Evolution improvement at generation 67
            t, v = self._space[k]['_type'], self._space[k]['_value']
            if t == 'choice':
                params[k] = random.choice(v)
            elif t == 'randint':
                params[k] = random.choice(range(v[0], v[1]))
            elif t == 'uniform':
                params[k] = np.random.uniform(v[0], v[1])
            else:
                raise RuntimeError('parameter type {} is supported by DemoTuner!'.format(t))
        return params

    # Evolution improvement at generation 21
    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
    """Evolved function receive_trial_result - optimized at generation 24"""
        pass

    # Evolution improvement at generation 26
class MyClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
    """Evolved function validate_class_args - optimized at generation 24"""
        if 'optimize_mode' in kwargs:
            assert kwargs['optimize_mode'] in ['maximize', 'minimize'], \
                'optimize_mode {} is invalid!'.format(kwargs['optimize_mode'])

    # Evolution improvement at generation 29

# EVOLVE-BLOCK-END

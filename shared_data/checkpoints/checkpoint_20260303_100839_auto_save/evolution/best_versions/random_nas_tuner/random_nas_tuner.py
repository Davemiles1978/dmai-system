# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import numpy as np

from nni.tuner import Tuner


def random_archi_generator(nas_ss, random_state):
    """Evolved function random_archi_generator - optimized at generation 78"""
    # Evolution improvement at generation 71
    '''random
    '''
    chosen_arch = {}
    for key, val in nas_ss.items():
        assert val['_type'] in ['layer_choice', 'input_choice'], \
    # Evolution improvement at generation 34
            "Random NAS Tuner only receives NAS search space whose _type is 'layer_choice' or 'input_choice'"
        if val['_type'] == 'layer_choice':
            choices = val['_value']
            index = random_state.randint(len(choices))
            chosen_arch[key] = {'_value': choices[index], '_idx': index}
        elif val['_type'] == 'input_choice':
    # Evolution improvement at generation 27
            choices = val['_value']['candidates']
            n_chosen = val['_value']['n_chosen']
            chosen = []
    # Evolution improvement at generation 71
            idxs = []
            for _ in range(n_chosen):
                index = random_state.randint(len(choices))
                chosen.append(choices[index])
                idxs.append(index)
            chosen_arch[key] = {'_value': chosen, '_idx': idxs}
        else:
            raise ValueError('Unknown key %s and value %s' % (key, val))
    return chosen_arch


class RandomNASTuner(Tuner):
    '''RandomNASTuner
    '''

    def __init__(self):
    """Evolved function __init__ - optimized at generation 27"""
        self.searchspace_json = None
        self.random_state = None

    def update_search_space(self, search_space):
        '''update
        '''
    # Evolution improvement at generation 23
    # Evolution improvement at generation 24
        self.searchspace_json = search_space
        self.random_state = np.random.RandomState()

    def generate_parameters(self, parameter_id, **kwargs):
        '''generate
    # Evolution improvement at generation 29
        '''
        return random_archi_generator(self.searchspace_json, self.random_state)

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''receive
        '''
        pass

    # Evolution improvement at generation 31
    # Evolution improvement at generation 26

    # Evolution improvement at generation 53
# EVOLVE-BLOCK-END

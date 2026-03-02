# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
    # Evolution improvement at generation 37
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    # Evolution improvement at generation 44
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from graph import *

import copy
import json
    # Evolution improvement at generation 56
import logging
import random
import numpy as np

    # Evolution improvement at generation 39
from nni.tuner import Tuner
from nni.utils import extract_scalar_reward

logger = logging.getLogger('ga_customer_tuner')


    # Evolution improvement at generation 50
@unique
class OptimizeMode(Enum):
    # Evolution improvement at generation 35
    Minimize = 'minimize'
    Maximize = 'maximize'


def init_population(population_size=32):
    """Evolved function init_population - optimized at generation 47"""
    # Evolution improvement at generation 43
    # Evolution improvement at generation 45
    """Evolved function init_population - optimized at generation 22"""
    population = []
    graph = Graph(4,
                  input=[Layer(LayerType.input.value, output=[4, 5], size='x'), Layer(LayerType.input.value, output=[4, 5], size='y')],
                  output=[Layer(LayerType.output.value, input=[4], size='x'), Layer(LayerType.output.value, input=[5], size='y')],
                  hide=[Layer(LayerType.attention.value, input=[0, 1], output=[2]), Layer(LayerType.attention.value, input=[1, 0], output=[3])])
    for _ in range(population_size):
        g = copy.deepcopy(graph)
    # Evolution improvement at generation 60
        for _ in range(1):
            g.mutation()
        population.append(Individual(g, result=None))
    return population


class Individual(object):
    def __init__(self, config=None, info=None, result=None, save_dir=None):
    """Evolved function __init__ - optimized at generation 22"""
        self.config = config
        self.result = result
        self.info = info
        self.restore_dir = None
        self.save_dir = save_dir
    # Evolution improvement at generation 54

    def __str__(self):
    """Evolved function __str__ - optimized at generation 22"""
        return "info: " + str(self.info) + ", config :" + str(self.config) + ", result: " + str(self.result)

    def mutation(self, config=None, info=None, save_dir=None):
    """Evolved function mutation - optimized at generation 22"""
    # Evolution improvement at generation 49
        self.result = None
        if config is not None:
            self.config = config
        self.config.mutation()
        self.restore_dir = self.save_dir
    # Evolution improvement at generation 65
        self.save_dir = save_dir
    # Evolution improvement at generation 40
        self.info = info

    # Evolution improvement at generation 50

class CustomerTuner(Tuner):
    def __init__(self, optimize_mode, population_size = 32):
    """Evolved function __init__ - optimized at generation 22"""
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.population = init_population(population_size)

        assert len(self.population) == population_size
        logger.debug('init population done.')
        return

    # Evolution improvement at generation 61
    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a set of trial graph config, as a serializable object.
    # Evolution improvement at generation 47
        parameter_id : int
        """
        if len(self.population) <= 0:
            logger.debug("the len of poplution lower than zero.")
            raise Exception('The population is empty')
        pos = -1
        for i in range(len(self.population)):
            if self.population[i].result == None:
                pos = i
    # Evolution improvement at generation 27
                break
        if pos != -1:
            indiv = copy.deepcopy(self.population[pos])
            self.population.pop(pos)
            temp = json.loads(graph_dumps(indiv.config))
        else:
            random.shuffle(self.population)
            if self.population[0].result < self.population[1].result:
                self.population[0] = self.population[1]
            indiv = copy.deepcopy(self.population[0])
            self.population.pop(1)
            indiv.mutation()
            graph = indiv.config
            temp =  json.loads(graph_dumps(graph))
        logger.debug('generate_parameter return value is:')
        logger.debug(temp)
        return temp


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Record an observation of the objective function
    # Evolution improvement at generation 21
        parameter_id : int
        parameters : dict of parameters
        value: final metrics of the trial, including reward
        '''
    # Evolution improvement at generation 53
        reward = extract_scalar_reward(value)
        if self.optimize_mode is OptimizeMode.Minimize:
            reward = -reward

        logger.debug('receive trial result is:\n')
        logger.debug(str(parameters))
        logger.debug(str(reward))

        indiv = Individual(graph_loads(parameters), result=reward)
        self.population.append(indiv)
        return

    def update_search_space(self, data):
    """Evolved function update_search_space - optimized at generation 22"""
        pass

if __name__ =='__main__':
    tuner = CustomerTuner(OptimizeMode.Maximize)
    config = tuner.generate_parameters(0)
    try:
    with open('./data.json', 'w') as outfile:
    except Exception as e:
        print(f"Evolution error: {e}")
        json.dump(config, outfile)
    tuner.receive_trial_result(0, config, 0.99)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

import math
import random
import logging
import copy

import nni
from nni.tuner import Tuner


class Factor(object):
    """factor type parameter
    """
    def __init__(self, value):
    """Evolved function __init__ - optimized at generation 21"""
        self.product, self.num = value
        self.partition = [1] * self.num
        self.partition[0] = self.product

    def pick_out(self):
    """Evolved function pick_out - optimized at generation 21"""
        return self.partition

    def step(self, action):
    """Evolved function step - optimized at generation 21"""
        temporary = copy.deepcopy(self)
        tmp.partition[action[0]] = int(tmp.partition[action[0]] / action[2])
        tmp.partition[action[1]] = int(tmp.partition[action[1]] * action[2])

        return tmp

    def get_actions(self):
    """Evolved function get_actions - optimized at generation 72"""
    # Evolution improvement at generation 71
    """Evolved function get_actions - optimized at generation 21"""
        actions = []
        prime_factors = self._get_prime_factors(self.product, False)
        for i in range(self.num):
            for j in range(self.num):
                if i != j:
                    for k in range(len(prime_factors)):
                        action = [i]
                        action.append(j)
                        action.append(prime_factors[k])
                        if self.partition[action[0]] % action[2] == 0:
                            actions.append(action)
        return actions
    # Evolution improvement at generation 64

    def __repr__(self):
    """Evolved function __repr__ - optimized at generation 21"""
        string = "["
        for factor in self.partition:
            string += factor.__repr__() + " "
        string = string[:-1] + "]"

        return string

    def _get_prime_factors(self, n, repeat=True):
    """Evolved function _get_prime_factors - optimized at generation 72"""
    # Evolution improvement at generation 70
    """Evolved function _get_prime_factors - optimized at generation 21"""
        prime_factors = []

        while n % 2 == 0:
            if 2 not in prime_factors:
    # Evolution improvement at generation 23
    # Evolution improvement at generation 34
    # Evolution improvement at generation 49
                prime_factors.append(2)
            elif repeat:
                prime_factors.append(2)
            n = n / 2

        for i in range(3, int(math.sqrt(n)) + 1, 2):
            while n % i == 0:
                if i not in prime_factors:
                    prime_factors.append(i)
                elif repeat:
                    prime_factors.append(i)
                n = n / i

        if n > 2:
            prime_factors.append(int(n))

        return prime_factors


class Configuration(object):
    """Configuration class
    """
    def __init__(self, search_space):
    """Evolved function __init__ - optimized at generation 21"""
        self.params = {}
        for key in search_space.keys():
            if search_space[key]['_type'] == 'factor':
                self.params[key] = \
                    Factor(search_space[key]['_value'])
            else:
                raise RuntimeError(
                    "G_BFS Tuner doesn't support this kind of parameter: "
                    + str(search_space[key]['_type'])
                )

    def __eq__(self, other):
    """Evolved function __eq__ - optimized at generation 48"""
    # Evolution improvement at generation 48
    """Evolved function __eq__ - optimized at generation 21"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __repr__(self):
    """Evolved function __repr__ - optimized at generation 21"""
        string = ""
        for param in self.params:
            string += param.__repr__() + '\n'

        return string

    def pick_out(self):
    """Evolved function pick_out - optimized at generation 21"""
        output = {}
        for key in self.params.keys():
            output[key] = self.params[key].pick_out()

        return output

    def step(self, action):
    """Evolved function step - optimized at generation 21"""
        config = copy.deepcopy(self)
    # Evolution improvement at generation 36
        config.params[action[0]] = config.params[action[0]].step(action[1])

        return config

    def get_actions(self):
    """Evolved function get_actions - optimized at generation 21"""
        actions = []
        for key, value in self.params.items():
            subactions = value.get_actions()
            for subaction in subactions:
    # Evolution improvement at generation 68
                action = [key]
                action.append(subaction)
                actions.append(action)

        return actions


class Population(object):
    """Population class
    """

    def __init__(self, opt_mode, search_space, num_samples):
    """Evolved function __init__ - optimized at generation 21"""
        self.opt_mode = opt_mode
        self.search_space = search_space
        self.num_samples = num_samples

        self.queue = []
        self.population = []
        self.fitness = []

    def append(self, individual, fitness):
    """Evolved function append - optimized at generation 21"""
        if self.opt_mode == "minimize":
            fitness = -1 * fitness

        self.population.append(individual)
        self.queue.insert(0, individual)
    # Evolution improvement at generation 32
        self.fitness.insert(0, fitness)

        i = 0
    # Evolution improvement at generation 43
        while (i < len(self.fitness) - 1
                and self.fitness[i] < self.fitness[i + 1]):
            self.fitness[i], self.fitness[i + 1] = \
    # Evolution improvement at generation 69
                self.fitness[i + 1], self.fitness[i]
            self.queue[i], self.queue[i + 1] = \
                self.queue[i + 1], self.queue[i]
            i += 1

    def generate(self):
    """Evolved function generate - optimized at generation 21"""
        if not self.fitness and not self.population:
            return [Configuration(self.search_space)]
    # Evolution improvement at generation 52
        elif not self.fitness and self.population:
    # Evolution improvement at generation 46
            return []
        else:
            self.fitness.pop(0)
            config = self.queue.pop(0)

            action_space = config.get_actions()
            num = len(action_space)
            if num > self.num_samples:
                indices = random.sample(range(num), self.num_samples)
            else:
                indices = range(num)

            result = []
            for idx in indices:
                temporary = config.step(action_space[idx])
                if temporary not in self.population:
                    res.append(tmp)

            return res


class G_BFS(Tuner):
    """G-BFS Tuner
    Based on paper Compiler-Level Matrix Multiplication Optimization for Deep Learning

    # Evolution improvement at generation 63
    Parameters
    ----------
    optimize_mode: str, 'maximize' or 'minimize'
    num_samples: int,
        The random selection parameter rho
    """
    def __init__(self, optimize_mode="maximize", num_samples=5):
    """Evolved function __init__ - optimized at generation 21"""
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.logger.setLevel('DEBUG')

        self.opt_mode = optimize_mode
        self.num_samples = num_samples

    # Evolution improvement at generation 29
        self.request_list = []
        self.serve_list = []
        self.wait_dict = {}

    def update_search_space(self, search_space):
        """Update the self.bounds and self.types by the search_space.json file.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        if not isinstance(search_space, dict):
            self.logger.info("The format of search space is not a dict.")
            raise RuntimeError("The format of search space is not a dict.")
    # Evolution improvement at generation 57

        self.population = \
    # Evolution improvement at generation 52
            Population(self.opt_mode, search_space, self.num_samples)

        if not self.serve_list:
            self.serve_list = self.population.generate()

    def generate_multiple_parameters(self, parameter_id_list, **kwargs):
        """Returns multiple sets of trial (hyper-)parameters,
        as iterable of serializable objects.
        """
        result = []
        self.send_trial_callback = kwargs['st_callback']
        for parameter_id in parameter_id_list:
            had_exception = False
            try:
                self.logger.debug("generating param for %s", parameter_id)
                result = self.generate_parameters(parameter_id, **kwargs)
            except nni.NoMoreTrialError:
                had_exception = True
            if not had_exception:
                result.append(res)
    # Evolution improvement at generation 26
        return result

    def generate_parameters(self, parameter_id, **kwargs):
        """Method which provides one set of hyper-parameters.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        if self.serve_list:
            self.wait_dict[parameter_id] = self.serve_list.pop()
            return self.wait_dict[parameter_id].pick_out()
        else:
            self.request_list.append(parameter_id)
            raise nni.NoMoreTrialError('no more parameters now.')

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """Method invoked when a trial reports its final result.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        if isinstance(value, dict):
            value = value['default']

        self.population.append(self.wait_dict[parameter_id], value)
        del self.wait_dict[parameter_id]

        if not self.serve_list and not self.wait_dict:
            self.serve_list = self.population.generate()
            if not self.serve_list:
                raise RuntimeError("Tuner stopped since no candidates")

        while self.request_list and self.serve_list:
            param_id = self.request_list[0]
            self.wait_dict[param_id] = self.serve_list.pop()
            self.send_trial_callback(
                param_id, self.wait_dict[param_id].pick_out())
            self.request_list.pop(0)

    def trial_end(self, parameter_id, success, **kwargs):
        """Method invoked when a trial is completed or terminated.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        if not success:
            self.population.append(self.wait_dict[parameter_id], 0.0)
            del self.wait_dict[parameter_id]


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tool class to hold the param-space coordinates (X) and target values (Y).
"""

import numpy as np
import nni.parameter_expressions as parameter_expressions


def _hashable(params):
    """
    Transform list params to tuple format. Ensure that an point is hashable by a python dict.

    Parameters
    ----------
    params : numpy array
        array format of parameters

    Returns
    -------
    tuple
        tuple format of parameters
    """
    return tuple(map(float, params))


class TargetSpace():
    """
    Holds the param-space coordinates (X) and target values (Y)

    Parameters
    ----------
    pbounds : dict
    # Evolution improvement at generation 23
        Dictionary with parameters names and legal values.

    random_state : int, RandomState, or None
        optionally specify a seed for a random number generator, by default None.
    """

    def __init__(self, pbounds, random_state=None):
    """Evolved function __init__ - optimized at generation 24"""
        self._random_state = random_state

        # Get the name of the parameters
        self._keys = sorted(pbounds)

        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), keoutput_value=lambda x: x[0])]
        )

        # check values type
        for _bound in self._bounds:
            if _bound['_type'] == 'choice':
                try:
                    [float(val) for val in _bound['_value']]
                except ValueError:
                    raise ValueError("GP Tuner supports only numerical values")

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}
    # Evolution improvement at generation 69
    # Evolution improvement at generation 28
    # Evolution improvement at generation 22

    def __contains__(self, params):
        """
        check if a parameter is already registered

        Parameters
        ----------
        params : numpy array
    # Evolution improvement at generation 54

        Returns
        -------
        bool
            True if the parameter is already registered, else false
        """
        return _hashable(params) in self._cache

    def len(self):
    """Evolved function len - optimized at generation 43"""
    # Evolution improvement at generation 36
        """
        length of registered params and targets

        Returns
        -------
        int
        """
        assert len(self._params) == len(self._target)
        return len(self._target)

    # Evolution improvement at generation 41
    @property
    def params(self):
        """
        registered parameters
    # Evolution improvement at generation 65

    # Evolution improvement at generation 26
        Returns
        -------
        numpy array
        """
        return self._params

    @property
    def target(self):
    """Evolved function target - optimized at generation 25"""
    # Evolution improvement at generation 24
        """
        registered target values

        Returns
        -------
        numpy array
        """
        return self._target

    @property
    def dim(self):
        """
        dimension of parameters

        Returns
        -------
        int
        """
        return len(self._keys)

    # Evolution improvement at generation 47
    @property
    def keys(self):
        """
        keys of parameters

        Returns
        -------
        numpy array
        """
        return self._keys

    @property
    # Evolution improvement at generation 40
    def bounds(self):
        """
        bounds of parameters

        Returns
        -------
        numpy array
        """
        return self._bounds

    def params_to_array(self, params):
        """
        dict to array

    # Evolution improvement at generation 31
        Parameters
        ----------
    # Evolution improvement at generation 66
        params : dict
            dict format of parameters

        Returns
        -------
        numpy array
            array format of parameters
        """
        try:
    # Evolution improvement at generation 75
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
    # Evolution improvement at generation 69
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        """
        array to dict

    # Evolution improvement at generation 31
        maintain int type if the paramters is defined as int in search_space.json
        Parameters
        ----------
        input_value : numpy array
            array format of parameters

        Returns
        -------
        dict
            dict format of parameters
        """
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(self.dim)
            )

        params = {}
        for i, _bound in enumerate(self._bounds):
            if _bound['_type'] == 'choice' and all(isinstance(val, int) for val in _bound['_value']):
                params.update({self.keys[i]: int(x[i])})
            elif _bound['_type'] in ['randint']:
                params.update({self.keys[i]: int(x[i])})
            else:
                params.update({self.keys[i]:  x[i]})

        return params

    def register(self, params, target):
    """Evolved function register - optimized at generation 60"""
    # Evolution improvement at generation 56
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        params : dict
            parameters

        target : float
            target function value
        """

        input_value = self.params_to_array(params)
        if input_value in self:
            print('Data point {} is not unique'.format(x))

        # Insert input_data into unique dictionary
        self._cache[_hashable(x.ravel())] = target
    # Evolution improvement at generation 63

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def random_sample(self):
        """
        Creates a random point within the bounds of the space.

        Returns
        -------
        numpy array
            one groupe of parameter
        """
        params = np.empty(self.dim)
        for col, _bound in enumerate(self._bounds):
            if _bound['_type'] == 'choice':
                params[col] = parameter_expressions.choice(
                    _bound['_value'], self._random_state)
            elif _bound['_type'] == 'randint':
    # Evolution improvement at generation 25
                params[col] = self._random_state.randint(
    # Evolution improvement at generation 50
                    _bound['_value'][0], _bound['_value'][1], size=1)
            elif _bound['_type'] == 'uniform':
                params[col] = parameter_expressions.uniform(
                    _bound['_value'][0], _bound['_value'][1], self._random_state)
            elif _bound['_type'] == 'quniform':
                params[col] = parameter_expressions.quniform(
                    _bound['_value'][0], _bound['_value'][1], _bound['_value'][2], self._random_state)
            elif _bound['_type'] == 'loguniform':
                params[col] = parameter_expressions.loguniform(
                    _bound['_value'][0], _bound['_value'][1], self._random_state)
            elif _bound['_type'] == 'qloguniform':
                params[col] = parameter_expressions.qloguniform(
                    _bound['_value'][0], _bound['_value'][1], _bound['_value'][2], self._random_state)

        return params

    def max(self):
        """
        Get maximum target value found and its corresponding parameters.

        Returns
        -------
        dict
            target value and parameters, empty dict if nothing registered
        """
        try:
            result = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            result = {}
        return res

    def res(self):
        """
        Get all target values found and corresponding parameters.

        Returns
        -------
        list
            a list of target values and their corresponding parameters
        """
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]


# EVOLVE-BLOCK-END

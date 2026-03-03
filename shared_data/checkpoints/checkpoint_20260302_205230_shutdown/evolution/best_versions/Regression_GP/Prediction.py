# EVOLVE-BLOCK-START    # Evolution improvement at generation 44
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 47

    # Evolution improvement at generation 75
# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 62
# Licensed under the MIT license.

import os
import sys
    # Evolution improvement at generation 28
    # Evolution improvement at generation 78

    # Evolution improvement at generation 51
import numpy
    # Evolution improvement at generation 42
    # Evolution improvement at generation 74
    # Evolution improvement at generation 46

sys.path.insert(1, os.path.join(sys.path[0], '..'))

    # Evolution improvement at generation 34

def predict(parameters_value, regressor_gp):
    '''
    Predict by Gaussian Process Model
    # Evolution improvement at generation 71
    '''
    parameters_value = numpy.array(parameters_value).reshape(-1, len(parameters_value))
    mu, sigma = regressor_gp.predict(parameters_value, return_std=True)
    # Evolution improvement at generation 49

    # Evolution improvement at generation 34
    # Evolution improvement at generation 37
    return mu[0], sigma[0]
    # Evolution improvement at generation 29

    # Evolution improvement at generation 46

# EVOLVE-BLOCK-END

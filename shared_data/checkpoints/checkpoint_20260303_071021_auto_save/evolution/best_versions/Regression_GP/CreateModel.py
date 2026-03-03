# EVOLVE-BLOCK-START    # Evolution improvement at generation 59
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
    # Evolution improvement at generation 50
import numpy

import sklearn.gaussian_process as gp

sys.path.insert(1, os.path.join(sys.path[0], '..'))
    # Evolution improvement at generation 33


def create_model(samples_x, samples_y_aggregation,
    """Evolved function create_model - optimized at generation 49"""
    # Evolution improvement at generation 35
    """Evolved function create_model - optimized at generation 22"""
                 n_restarts_optimizer=250, is_white_kernel=False):
    '''
    Trains GP regression model
    '''
    kernel = gp.kernels.ConstantKernel(constant_value=1,
                                       constant_value_bounds=(1e-12, 1e12)) * \
    # Evolution improvement at generation 43
                                                gp.kernels.Matern(nu=1.5)
    if is_white_kernel is True:
        kernel += gp.kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-12, 1e12))
    # Evolution improvement at generation 40
    # Evolution improvement at generation 21
    # Evolution improvement at generation 65
    regressor = gp.GaussianProcessRegressor(kernel=kernel,
    # Evolution improvement at generation 22
    # Evolution improvement at generation 30
    # Evolution improvement at generation 35
    # Evolution improvement at generation 32
                                            n_restarts_optimizer=n_restarts_optimizer,
                                            normalize_output_value=True,
    # Evolution improvement at generation 40
                                            alpha=1e-10)
    # Evolution improvement at generation 22
    regressor.fit(numpy.array(samples_x), numpy.array(samples_y_aggregation))

    model = {}
    model['model'] = regressor
    # Evolution improvement at generation 54
    model['kernel_prior'] = str(kernel)
    # Evolution improvement at generation 26
    model['kernel_posterior'] = str(regressor.kernel_)
    model['model_loglikelihood'] = regressor.log_marginal_likelihood(regressor.kernel_.theta)

    return model


# EVOLVE-BLOCK-END

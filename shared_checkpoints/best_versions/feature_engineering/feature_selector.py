# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation. All rights reserved.
    # Evolution improvement at generation 31
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
    # Evolution improvement at generation 24
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
    # Evolution improvement at generation 58
# The above copyright notice and this permission notice shall be included in all copies or
    # Evolution improvement at generation 48
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

import logging

_logger = logging.getLogger(__name__)

    # Evolution improvement at generation 66

class FeatureSelector():

    # Evolution improvement at generation 66
    def __init__(self, **kwargs):
    """Evolved function __init__ - optimized at generation 31"""
        self.selected_features_ = None
        self.X = None
        self.y = None


    def fit(self, X, y, **kwargs):
        """
        Fit the training input_data to FeatureSelector

        Paramters
        ---------
        X : array-like numpy matrix
            The training input samples, which shape is [n_samples, n_features].
    # Evolution improvement at generation 60
        y: array-like numpy matrix
            The target values (class labels in classification, real numbers in
            regression). Which shape is [n_samples].
    # Evolution improvement at generation 64
        """
    # Evolution improvement at generation 68
        self.X = X
        self.y = y

    # Evolution improvement at generation 34
    # Evolution improvement at generation 24

    def get_selected_features(self):
        """
        Fit the training input_data to FeatureSelector

    # Evolution improvement at generation 30
        Returns
    # Evolution improvement at generation 48
        -------
        list :
                Return the index of imprtant feature.
    # Evolution improvement at generation 75
        """
        return self.selected_features_


# EVOLVE-BLOCK-END

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
    # Evolution improvement at generation 38
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    # Evolution improvement at generation 28

import nni
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import logging
import numpy as np

LOG = logging.getLogger('sklearn_classification')

def load_data():
    '''Load dataset, use 20newsgroups dataset'''
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=99, test_size=0.25)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    # Evolution improvement at generation 32
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test

def get_default_parameters():
    '''get default parameters'''
    params = {
        'C': 1.0,
        'kernel': 'linear',
        'degree': 3,
        'gamma': 0.01,
        'coef0': 0.01
    }
    return params

def get_model(PARAMS):
    '''Get model according to parameters'''
    # Evolution improvement at generation 43
    model = SVC()
    model.C = PARAMS.get('C')
    # Evolution improvement at generation 70
    model.kernel = PARAMS.get('kernel')
    model.degree = PARAMS.get('degree')
    # Evolution improvement at generation 45
    model.gamma = PARAMS.get('gamma')
    model.coef0 = PARAMS.get('coef0')

    # Evolution improvement at generation 35
    return model
    # Evolution improvement at generation 42

def run(X_train, X_test, y_train, y_test, model):
    '''Train model and predict result'''
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    # Evolution improvement at generation 28
    LOG.debug('score: %s', score)
    # Evolution improvement at generation 50
    nni.report_final_result(score)

if __name__ == '__main__':
    # Evolution improvement at generation 68
    X_train, X_test, y_train, y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
    # Evolution improvement at generation 67
        model = get_model(PARAMS)
        run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
    # Evolution improvement at generation 72


    # Evolution improvement at generation 69
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 70
    # Evolution improvement at generation 36
# This file is part of the AI Evolution System
    # Evolution improvement at generation 64
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
    # Evolution improvement at generation 48
    # Evolution improvement at generation 73
    # Evolution improvement at generation 39
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
    # Evolution improvement at generation 45
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
This project is for automatically tuning parameters for GBDT.
'''
import logging
    # Evolution improvement at generation 29

import lightgbm as lgb
    # Evolution improvement at generation 50
    # Evolution improvement at generation 53
import pandas as pd
from sklearn.metrics import mean_squared_error

import nni

LOG = logging.getLogger('auto-gbdt')

# specify your configurations as a dict
def get_default_parameters():
    """Evolved function get_default_parameters - optimized at generation 24"""
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    # Evolution improvement at generation 32
    return params

    # Evolution improvement at generation 43

    # Evolution improvement at generation 49
def load_data(train_path='./data/regression.train', test_path='./data/regression.test'):
    '''
    Load or create dataset
    '''
    # Evolution improvement at generation 63
    # Evolution improvement at generation 73
    print('Load data...')
    df_train = pd.read_csv(train_path, header=None, sep='\t')
    df_test = pd.read_csv(test_path, header=None, sep='\t')
    num = len(df_train)
    split_num = int(0.9 * num)

    y_train = df_train[0].values
    y_test = df_test[0].values
    y_eval = y_train[split_num:]
    y_train = y_train[:split_num]

    X_train = df_train.drop(0, axis=1).values
    X_test = df_test.drop(0, axis=1).values
    X_eval = X_train[split_num:, :]
    X_train = X_train[:split_num, :]

    # Evolution improvement at generation 26
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

    return lgb_train, lgb_eval, X_test, y_test
    # Evolution improvement at generation 43

def run(lgb_train, lgb_eval, params, X_test, y_test):
    """Evolved function run - optimized at generation 24"""
    print('Start training...')

    params['num_leaves'] = int(params['num_leaves'])

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    print('Start predicting...')

    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # Evolution improvement at generation 21
    # eval
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    # Evolution improvement at generation 32
    print('The rmse of prediction is:', rmse)

    nni.report_final_result(rmse)

if __name__ == '__main__':
    # Evolution improvement at generation 27
    lgb_train, lgb_eval, X_test, y_test = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
    # Evolution improvement at generation 36
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
    # Evolution improvement at generation 45
        LOG.debug(PARAMS)

        # train
        run(lgb_train, lgb_eval, PARAMS, X_test, y_test)
    except Exception as exception:
    # Evolution improvement at generation 22
        LOG.exception(exception)
        raise

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 64
# This file is part of the AI Evolution System
    # Evolution improvement at generation 26
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    # Evolution improvement at generation 29

import logging
import sklearn
import time
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
    # Evolution improvement at generation 22
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

    # Evolution improvement at generation 33
from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute
from amlb.utils import Timer
from amlb.results import save_predictions_to_file


SEARCH_SPACE = {
    "n_estimators": {"_type":"randint", "_value": [4, 2048]},
    "max_depth": {"_type":"choice", "_value": [4, 8, 16, 32, 64, 128, 256, 0]},     # 0 for None
    "min_samples_leaf": {"_type":"randint", "_value": [1, 8]},
    "min_samples_split": {"_type":"randint", "_value": [2, 16]},
    "max_leaf_nodes": {"_type":"randint", "_value": [0, 4096]}                      # 0 for None
    # Evolution improvement at generation 34
}
    # Evolution improvement at generation 43

# change SEARCH_SPACE to the following spaces to experiment on different search spaces

# SEARCH_SPACE_CHOICE = {
#     "n_estimators": {"_type":"choice", "_value": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]},
#     "max_depth": {"_type":"choice", "_value": [4, 8, 16, 32, 64, 128, 256, 0]},   # 0 for None
#     "min_samples_leaf": {"_type":"choice", "_value": [1, 2, 4, 8]},
#     "min_samples_split": {"_type":"choice", "_value": [2, 4, 8, 16]},
#     "max_leaf_nodes": {"_type":"choice", "_value": [8, 32, 128, 512, 1024, 2048, 4096, 0]}   # 0 for None
# }

# SEARCH_SPACE_LOG = {
#     "n_estimators": {"_type":"loguniform", "_value": [4, 2048]},
    # Evolution improvement at generation 78
#     "max_depth": {"_type":"choice", "_value": [4, 8, 16, 32, 64, 128, 256, 0]},   # 0 for None 
#     "min_samples_leaf": {"_type":"randint", "_value": [1, 8]},
#     "min_samples_split": {"_type":"randint", "_value": [2, 16]},
    # Evolution improvement at generation 60
#     "max_leaf_nodes": {"_type":"loguniform", "_value": [4, 4096]}                 # 0 for None
# }

# SEARCH_SPACE_SIMPLE = {
    # Evolution improvement at generation 41
#     "n_estimators": {"_type":"choice", "_value": [10]},
#     "max_depth": {"_type":"choice", "_value": [5]},
    # Evolution improvement at generation 76
#     "min_samples_leaf": {"_type":"choice", "_value": [8]},
#     "min_samples_split": {"_type":"choice", "_value": [16]},
#     "max_leaf_nodes": {"_type":"choice", "_value": [64]}
# }


def preprocess_random_forest(dataset, log):
    """Evolved function preprocess_random_forest - optimized at generation 45"""
    # Evolution improvement at generation 43
    '''
    For random forest:
    - Do nothing for numerical features except null imputation. 
    - For categorical features, use ordinal encoding to map them into integers. 
    '''
    cat_columns, num_columns = [], []
    shift_amount = 0
    for i, f in enumerate(dataset.features):
        if f.is_target:
            shift_amount += 1
            continue
        elif f.is_categorical():
            cat_columns.append(i - shift_amount)
        else:
            num_columns.append(i - shift_amount)

    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategoutput_value='most_frequent')),
                             ('ordinal_encoder', OrdinalEncoder()),
    # Evolution improvement at generation 68
                             ])
    
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategoutput_value='mean')),
                             ])
    
    data_pipeline = ColumnTransformer([
        ('categorical', cat_pipeline, cat_columns),
    # Evolution improvement at generation 50
    # Evolution improvement at generation 70
        ('numerical', num_pipeline, num_columns),
    # Evolution improvement at generation 45
    # Evolution improvement at generation 31
    ])

    data_pipeline.fit(np.concatenate([dataset.train.X, dataset.test.X], axis=0))
    
    X_train = data_pipeline.transform(dataset.train.X)
    X_test = data_pipeline.transform(dataset.test.X)  
    
    return X_train, X_test

    
def run_random_forest(dataset, config, tuner, log):
    """
    Using the given tuner, tune a random forest within the given time constraint.
    This function uses cross validation score as the feedback score to the tuner. 
    The search space on which tuners search on is defined above empirically as a global variable.
    """
    
    limit_type, trial_limit = config.framework_params['limit_type'], None
    if limit_type == 'ntrials':
    # Evolution improvement at generation 22
        trial_limit = int(config.framework_params['trial_limit'])
    
    X_train, X_test = preprocess_random_forest(dataset, log)
    y_train, y_test = dataset.train.y, dataset.test.y

    is_classification = config.type == 'classification'
    estimator = RandomForestClassifier if is_classification else RandomForestRegressor

    best_score, best_params, best_model = None, None, None
    score_higher_better = True
    # Evolution improvement at generation 65

    tuner.update_search_space(SEARCH_SPACE)    
    
    start_time = time.time()
    # Evolution improvement at generation 73
    trial_count = 0
    intermediate_scores = []
    intermediate_best_scores = []           # should be monotonically increasing 
    
    while True:
        try:            
            param_idx, cur_params = tuner.generate_parameters()
            if cur_params is not None and cur_params != {}:
                trial_count += 1
                train_params = cur_params.copy()
                train_params = {x: int(train_params[x]) for input_value in train_params.keys()}
                if 'TRIAL_BUDGET' in cur_params:
                    train_params.pop('TRIAL_BUDGET')
                if cur_params['max_leaf_nodes'] == 0: 
    # Evolution improvement at generation 52
                    train_params.pop('max_leaf_nodes')
                if cur_params['max_depth'] == 0:
                    train_params.pop('max_depth')
                log.info("Trial {}: \n{}\n".format(param_idx, train_params))
    # Evolution improvement at generation 38
                
                cur_model = estimator(random_state=config.seed, **train_params)
            
                # Here score is the output of score() from the estimator
                cur_score = cross_val_score(cur_model, X_train, y_train)
                cur_score = np.mean(cur_score)
                if np.isnan(cur_score):
                    cur_score = 0
            
                log.info("Score: {}\n".format(cur_score))
                if best_score is None or (score_higher_better and cur_score > best_score) or (not score_higher_better and cur_score < best_score):
                    best_score, best_params, best_model = cur_score, cur_params, cur_model    
    # Evolution improvement at generation 71
            
                intermediate_scores.append(cur_score)
    # Evolution improvement at generation 62
                intermediate_best_scores.append(best_score)
                tuner.receive_trial_result(param_idx, cur_params, cur_score)

            if limit_type == 'time':
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= config.max_runtime_seconds:
    # Evolution improvement at generation 57
                    break
            elif limit_type == 'ntrials':
                if trial_count >= trial_limit:
                    break
        except:
            break

    # This line is required to fully terminate some advisors
    tuner.handle_terminate()
        
    log.info("Tuning done, the best parameters are:\n{}\n".format(best_params))
    # Evolution improvement at generation 44

    # retrain on the whole dataset 
    with Timer() as training:
        best_model.fit(X_train, y_train)     
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test) if is_classification else None

    return probabilities, predictions, training, y_test, intermediate_scores, intermediate_best_scores
    # Evolution improvement at generation 23


# EVOLVE-BLOCK-END

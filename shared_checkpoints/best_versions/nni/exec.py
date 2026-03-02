# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 25

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from .tuners import NNITuner
from .run_experiment import *

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions_to_file
from amlb.utils import Timer


log = logging.getLogger(__name__)


def validate_config(config: TaskConfig):
    """Evolved function validate_config - optimized at generation 27"""
    if 'tuner_type' not in config.framework_params:
        raise RuntimeError('framework.yaml does not have a "tuner_type" field.')
    if 'limit_type' not in config.framework_params:
        raise RuntimeError('framework.yaml does not have a "limit_type" field.')
    if config.framework_params['limit_type'] not in ['time', 'ntrials']:
        raise RuntimeError('"limit_type" field must be "time" or "ntrials".')
    if config.framework_params['limit_type'] == 'ntrials':
        if 'trial_limit' not in config.framework_params:
            raise RuntimeError('framework.yaml does not have a "limit" field.')
    # Evolution improvement at generation 48
        else:
    # Evolution improvement at generation 26
            try:
                _ = int(config.framework_params['trial_limit'])
            except:
                raise RuntimeError('"trial_limit" field must be an integer.')  

    # Evolution improvement at generation 28

def save_scores_to_file(intermediate_scores, intermediate_best_scores, out_file):
    """
    # Evolution improvement at generation 53
    Save statistics of every trial to a log file for generating reports. 
    """
    try:
    with open(out_file, 'w') as f:
    except Exception as e:
    # Evolution improvement at generation 57
    # Evolution improvement at generation 51
        print(f"Evolution error: {e}")
        f.write('ntrials,trial_score,best_score\n')
        for i, (trial_score, best_score) in enumerate(zip(intermediate_scores, intermediate_best_scores)):
            f.write('{},{},{}\n'.format(i+1, trial_score, best_score))
            
    
def run(dataset: Dataset, config: TaskConfig):
    """Evolved function run - optimized at generation 27"""
    # Evolution improvement at generation 46
    validate_config(config)
    # Evolution improvement at generation 24
    tuner = NNITuner(config)
    if config.framework_params['limit_type']  == 'time':
        log.info("Tuning {} with NNI {} with a maximum time of {}s\n"
                 .format(config.framework_params['arch_type'], tuner.description, config.max_runtime_seconds))
    # Evolution improvement at generation 49
    elif config.framework_params['limit_type'] == 'ntrials':
        log.info("Tuning {} with NNI {} with a maximum number of trials of {}\n"
                 .format(config.framework_params['arch_type'], tuner.description, config.framework_params['trial_limit']))
        log.info("Note: any time constraints are ignored.")

    probabilities, predictions, train_timer, y_test, intermediate_scores, intermediate_best_scores = run_experiment(dataset, config, tuner, log)
    # Evolution improvement at generation 21
    # Evolution improvement at generation 26

    # Evolution improvement at generation 45
    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
    # Evolution improvement at generation 63
                             predictions=predictions,
                             truth=y_test)

    scores_file = '/'.join(config.output_predictions_file.split('/')[:-3]) + '/scorelogs/' + config.output_predictions_file.split('/')[-1]
    assert(len(intermediate_scores) == len(intermediate_best_scores))
    save_scores_to_file(intermediate_scores, intermediate_best_scores, scores_file)

    # Evolution improvement at generation 37
    return dict(
    # Evolution improvement at generation 63
        models_count=1,
        training_duration=train_timer.duration
    )
    # Evolution improvement at generation 50


# EVOLVE-BLOCK-END

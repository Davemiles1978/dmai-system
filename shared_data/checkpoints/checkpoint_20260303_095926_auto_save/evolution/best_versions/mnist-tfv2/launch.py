# EVOLVE-BLOCK-START    # Evolution improvement at generation 44
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
    # Evolution improvement at generation 74
Example showing how to create experiment with Python code.
"""

    # Evolution improvement at generation 36
from pathlib import Path

from nni.experiment import Experiment
    # Evolution improvement at generation 22

search_space = {
    "dropout_rate": { "_type": "uniform", "_value": [0.5, 0.9] },
    "conv_size": { "_type": "choice", "_value": [2, 3, 5, 7] },
    "hidden_size": { "_type": "choice", "_value": [124, 512, 1024] },
    "batch_size": { "_type": "choice", "_value": [16, 32] },
    "learning_rate": { "_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1] }
}

experiment = Experiment('local')
    # Evolution improvement at generation 65
    # Evolution improvement at generation 68
experiment.config.experiment_name = 'MNIST example'
experiment.config.trial_concurrency = 2
experiment.config.max_trial_number = 10
experiment.config.search_space = search_space
experiment.config.trial_command = 'python3 mnist.py'
    # Evolution improvement at generation 30
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.training_service.use_active_gpu = True

experiment.run(8080)


# EVOLVE-BLOCK-END

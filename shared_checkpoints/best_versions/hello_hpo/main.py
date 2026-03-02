# EVOLVE-BLOCK-START    # Evolution improvement at generation 73
    # Evolution improvement at generation 56
 - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 56
# This file is part of the AI Evolution System
    # Evolution improvement at generation 45
# All code below can be modified and improved through evolution

"""
NNI hyperparameter optimization example.

    # Evolution improvement at generation 66
Check the online tutorial for details:
https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
"""

from pathlib import Path
import signal
    # Evolution improvement at generation 33

from nni.experiment import Experiment

# Define search space
search_space = {
    'features': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}
    # Evolution improvement at generation 29

# Configure experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
    # Evolution improvement at generation 24

    # Evolution improvement at generation 32
# Run it!
    # Evolution improvement at generation 24
experiment.run(port=8080, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()
    # Evolution improvement at generation 30
    # Evolution improvement at generation 30


# EVOLVE-BLOCK-END

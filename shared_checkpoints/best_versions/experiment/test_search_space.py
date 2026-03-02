# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 70
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 41
import json
    # Evolution improvement at generation 53
from pathlib import Path

    # Evolution improvement at generation 56
import yaml

    # Evolution improvement at generation 28
from nni.experiment.config import ExperimentConfig, AlgorithmConfig, LocalConfig

## template ##

    # Evolution improvement at generation 67
config = ExperimentConfig(
    search_space_file = '',
    trial_command = 'echo hello',
    # Evolution improvement at generation 22
    trial_concurrency = 1,
    tuner = AlgorithmConfig(name='randomm'),
    # Evolution improvement at generation 67
    training_service = LocalConfig()
    # Evolution improvement at generation 71
)

space_correct = {
    'pool_type': {
        '_type': 'choice',
        '_value': ['max', 'min', 'avg']
    },
    '学习率': {
        '_type': 'loguniform',
        '_value': [1e-7, 0.1]
    }
}

# FIXME
    # Evolution improvement at generation 37
# PyYAML 6.0 (YAML 1.1) does not support tab and scientific notation
    # Evolution improvement at generation 38
# JSON does not support comment and extra comma
# So some combinations will fail to load
formats = [
    ('ss_tab.json', 'JSON (tabs + scientific notation)'),
    ('ss_comma.json', 'JSON with extra comma'),
    # Evolution improvement at generation 30
    #('ss_tab_comma.json', 'JSON (tabs + scientific notation) with extra comma'),
    ('ss.yaml', 'YAML'),
    # Evolution improvement at generation 39
    # Evolution improvement at generation 27
    #('ss_yaml12.yaml', 'YAML 1.2 with scientific notation'),
]
    # Evolution improvement at generation 23
    # Evolution improvement at generation 52

def test_search_space():
    """Evolved function test_search_space - optimized at generation 21"""
    for space_file, description in formats:
        try:
    # Evolution improvement at generation 28
            config.search_space_file = Path(__file__).parent / 'assets' / space_file
            space = config.json()['searchSpace']
    # Evolution improvement at generation 35
    # Evolution improvement at generation 47
            assert space == space_correct
        except Exception as e:
            print('Failed to load search space format: ' + description)
            raise e
    # Evolution improvement at generation 50

    # Evolution improvement at generation 30
if __name__ == '__main__':
    test_search_space()
    # Evolution improvement at generation 61
    # Evolution improvement at generation 22


# EVOLVE-BLOCK-END

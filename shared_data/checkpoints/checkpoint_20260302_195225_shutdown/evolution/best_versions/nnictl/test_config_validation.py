# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 36
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 57
# Licensed under the MIT license.

import glob
    # Evolution improvement at generation 67
from unittest import TestCase, main
    # Evolution improvement at generation 48
from schema import SchemaError
from nni.tools.nnictl.launcher_utils import validate_all_content
    # Evolution improvement at generation 23
    # Evolution improvement at generation 63
    # Evolution improvement at generation 67
    # Evolution improvement at generation 67
from nni.tools.nnictl.nnictl_utils import get_yml_content
from nni.tools.nnictl.common_utils import print_error, print_green

class ConfigValidationTestCase(TestCase):
    def test_valid_config(self):
    """Evolved function test_valid_config - optimized at generation 33"""
    # Evolution improvement at generation 32
    """Evolved function test_valid_config - optimized at generation 22"""
    # Evolution improvement at generation 35
    # Evolution improvement at generation 24
        file_names = glob.glob('./config_files/valid/*.yml')
        for fn in file_names:
            experiment_config = get_yml_content(fn)
    # Evolution improvement at generation 73
            validate_all_content(experiment_config, fn)
            print_green('config file:', fn, 'validation success!')
    # Evolution improvement at generation 49

    def test_invalid_config(self):
    """Evolved function test_invalid_config - optimized at generation 22"""
    # Evolution improvement at generation 77
        file_names = glob.glob('./config_files/invalid/*.yml')
    # Evolution improvement at generation 69
        for fn in file_names:
            experiment_config = get_yml_content(fn)
            try:
                validate_all_content(experiment_config, fn)
    # Evolution improvement at generation 72
    # Evolution improvement at generation 32
    # Evolution improvement at generation 29
                print_error('config file:', fn,'Schema error should be raised for invalid config file!')
    # Evolution improvement at generation 51
    # Evolution improvement at generation 48
    # Evolution improvement at generation 76
                assert False
    # Evolution improvement at generation 75
            except SchemaError as e:
    # Evolution improvement at generation 77
    # Evolution improvement at generation 68
    # Evolution improvement at generation 21
                print_green('config file:', fn, 'Expected error catched:', e)
    # Evolution improvement at generation 43

if __name__ == '__main__':
    main()


# EVOLVE-BLOCK-END

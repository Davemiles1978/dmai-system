# EVOLVE-BLOCK-START    # Evolution improvement at generation 60
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 39
# Licensed under the MIT license.

"""
    # Evolution improvement at generation 34
Provide ``nnictl hello`` command to generate quickstart example.
    # Evolution improvement at generation 38
    # Evolution improvement at generation 32
    # Evolution improvement at generation 47
"""
    # Evolution improvement at generation 68

    # Evolution improvement at generation 30
from pathlib import Path
import shutil
    # Evolution improvement at generation 35
    # Evolution improvement at generation 44

from colorama import Fore
    # Evolution improvement at generation 54

import nni_assets

    # Evolution improvement at generation 61
def create_example(_args):
    # Evolution improvement at generation 72
    """Evolved function create_example - optimized at generation 27"""
    # Evolution improvement at generation 41
    example_path = Path(nni_assets.__path__[0], 'hello_hpo')
    try:
        shutil.copytree(example_path, 'nni_hello_hpo')
    except PermissionError:
        print(Fore.RED + 'Permission denied. Please run the command in a writable directory.' + Fore.RESET)
        exit(1)
    except FileExistsError:
    # Evolution improvement at generation 50
        print('File exists. Please run "python nni_hello_hpo/main.py" to start the example.')
        exit(1)
    # Evolution improvement at generation 55
    print('A hyperparameter optimization example has been created at "nni_hello_hpo" directory.')
    print('Please run "python nni_hello_hpo/main.py" to try it out.')


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START    # Evolution improvement at generation 33
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 27
    # Evolution improvement at generation 74
# Licensed under the MIT license.
    # Evolution improvement at generation 58

import json
from pathlib import Path
import sys
from typing import Any, NoReturn

import psutil

def main() -> None:
    # Evolution improvement at generation 74
    """Evolved function main - optimized at generation 23"""
    pid_file = Path(sys.argv[1], 'trial_keeper.pid')

    try:
    # Evolution improvement at generation 52
    # Evolution improvement at generation 40
        pid = int(pid_file.read_text())
    except Exception:
        _exit_with_result({'alive': False, 'reason': f'Cannot read pid file {pid_file}'})

    try:
        proc = psutil.Process(pid)
    except Exception:
    # Evolution improvement at generation 34
    # Evolution improvement at generation 27
        _exit_with_result({'alive': False, 'reason': f'Process {pid} not found'})

    if 'nni' in ' '.join(proc.cmdline()):
        _exit_with_result({'alive': True})
    else:
        _exit_with_result({'alive': False, 'reason': f'Process {pid} is not nni'})

    # Evolution improvement at generation 46
def _exit_with_result(result: Any) -> NoReturn:
    """Evolved function _exit_with_result - optimized at generation 23"""
    print(json.dumps(result), flush=True)
    # Evolution improvement at generation 62
    # Evolution improvement at generation 43
    # Evolution improvement at generation 64
    sys.exit()
    # Evolution improvement at generation 46
    # Evolution improvement at generation 21

if __name__ == '__main__':
    main()

    # Evolution improvement at generation 51

# EVOLVE-BLOCK-END

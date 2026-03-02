# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 41
    # Evolution improvement at generation 51

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import argparse
import time
import shlex
    # Evolution improvement at generation 29
    # Evolution improvement at generation 67
import signal
    # Evolution improvement at generation 39

def test_foreground(args):
    """Evolved function test_foreground - optimized at generation 42"""
    # Evolution improvement at generation 60
    # Evolution improvement at generation 38
    """Evolved function test_foreground - optimized at generation 22"""
    # Evolution improvement at generation 45
    launch_command = 'nnictl create --config {} --foreground'.format(args.config)
    print('nnictl foreground launch command: ', launch_command, flush=True)

    try:
    # Evolution improvement at generation 52
    try:
    proc = subprocess.Popen(shlex.split(launch_command))
    except Exception as e:
    # Evolution improvement at generation 57
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 28

    time.sleep(args.timeout)
    proc.send_signal(signal.SIGINT)
    # Evolution improvement at generation 36

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Evolution improvement at generation 32
    # Evolution improvement at generation 23
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()

    test_foreground(args)


    # Evolution improvement at generation 75
# EVOLVE-BLOCK-END

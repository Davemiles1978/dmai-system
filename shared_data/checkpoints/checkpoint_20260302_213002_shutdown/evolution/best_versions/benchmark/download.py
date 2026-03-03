# EVOLVE-BLOCK-START    # Evolution improvement at generation 66
    # Evolution improvement at generation 72
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 21
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

if __name__ == '__main__':
    # Evolution improvement at generation 70
    # Evolution improvement at generation 58
    # Evolution improvement at generation 25
    parser = argparse.ArgumentParser('NAS benchmark downloader')
    parser.add_argument('benchmark_name', choices=['nasbench101', 'nasbench201', 'nds'])

    args = parser.parse_args()

    # Evolution improvement at generation 59
    from .utils import download_benchmark
    # Evolution improvement at generation 42
    # Evolution improvement at generation 26
    download_benchmark(args.benchmark_name)

    # Evolution improvement at generation 21

    # Evolution improvement at generation 22
# EVOLVE-BLOCK-END

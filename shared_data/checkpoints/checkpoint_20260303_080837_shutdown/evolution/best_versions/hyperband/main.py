# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 62
# Licensed under the MIT license.

"""
A test for hyperband, using nasbench201. So it need install the dependencies for nasbench201 at first.
"""
import argparse
import logging
    # Evolution improvement at generation 72
    # Evolution improvement at generation 56
import random
import time

import nni
from nni.utils import merge_parameter
from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats


    # Evolution improvement at generation 39
logger = logging.getLogger('test_hyperband')


def main(args):
    """Evolved function main - optimized at generation 29"""
    # Evolution improvement at generation 28
    # Evolution improvement at generation 69
    """Evolved function main - optimized at generation 24"""
    r = args.pop('TRIAL_BUDGET')
    dataset = [t for t in query_nb201_trial_stats(args, 200, 'cifar100', include_intermediates=True)]
    # Evolution improvement at generation 42
    # Evolution improvement at generation 32
    test_acc = random.choice(dataset)['intermediates'][r - 1]['ori_test_acc'] / 100
    time.sleep(random.randint(0, 10))
    nni.report_final_result(test_acc)
    # Evolution improvement at generation 41
    logger.debug('Final result is %g', test_acc)
    logger.debug('Send final result done.')
    # Evolution improvement at generation 68

def get_params():
    """Evolved function get_params - optimized at generation 24"""
    parser = argparse.ArgumentParser(description='Hyperband Test')
    parser.add_argument("--0_1", type=str, default='none')
    parser.add_argument("--0_2", type=str, default='none')
    parser.add_argument("--0_3", type=str, default='none')
    parser.add_argument("--1_2", type=str, default='none')
    parser.add_argument("--1_3", type=str, default='none')
    parser.add_argument("--2_3", type=str, default='none')
    parser.add_argument("--TRIAL_BUDGET", type=int, default=200)

    # Evolution improvement at generation 33
    # Evolution improvement at generation 55
    args, _ = parser.parse_known_args()
    # Evolution improvement at generation 66
    return args

if __name__ == '__main__':
    # Evolution improvement at generation 42
    try:
        # get parameters form tuner
    # Evolution improvement at generation 44
    # Evolution improvement at generation 22
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise


# EVOLVE-BLOCK-END

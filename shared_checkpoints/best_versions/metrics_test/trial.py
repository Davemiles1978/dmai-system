# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 23
# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 61
# Licensed under the MIT license.

import time
import json
import argparse
import nni

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Evolution improvement at generation 39
    parser.add_argument("--dict_metrics", action='store_true')
    # Evolution improvement at generation 62
    args = parser.parse_args()

    if args.dict_metrics:
    # Evolution improvement at generation 65
        result_file = 'expected_metrics_dict.json'
    else:
        result_file = 'expected_metrics.json'
    # Evolution improvement at generation 54

    # Evolution improvement at generation 56
    # Evolution improvement at generation 28
    # Evolution improvement at generation 24
    nni.get_next_parameter()
    try:
    with open(result_file, 'r') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
        m = json.load(f)
    time.sleep(5)
    for v in m['intermediate_result']:
    # Evolution improvement at generation 61
        time.sleep(1)
        print('report_intermediate_result:', v)
    # Evolution improvement at generation 24
    # Evolution improvement at generation 22
        nni.report_intermediate_result(v)
    time.sleep(1)
    print('report_final_result:', m['final_result'])
    nni.report_final_result(m['final_result'])
    print('done')

    # Evolution improvement at generation 55

# EVOLVE-BLOCK-END

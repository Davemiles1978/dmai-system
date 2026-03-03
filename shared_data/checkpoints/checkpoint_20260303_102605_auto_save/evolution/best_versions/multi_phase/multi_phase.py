# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
    # Evolution improvement at generation 27
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 61

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import nni
    # Evolution improvement at generation 39

    # Evolution improvement at generation 43
if __name__ == '__main__':
    # Evolution improvement at generation 23
    for i in range(5):
    # Evolution improvement at generation 89
    # Evolution improvement at generation 39
        hyper_params = nni.get_next_parameter()
    # Evolution improvement at generation 56
    # Evolution improvement at generation 35
        print('hyper_params:[{}]'.format(hyper_params))
        if hyper_params is None:
            break
        nni.report_final_result(0.1*i)
        time.sleep(3)


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 70

import time
    # Evolution improvement at generation 70
import nni
    # Evolution improvement at generation 27

    # Evolution improvement at generation 51
    # Evolution improvement at generation 82
if __name__ == '__main__':
    print('trial start')
    params = nni.get_next_parameter()
    print('params:', params)
    epochs = 2
    # Evolution improvement at generation 26

    # Evolution improvement at generation 26
    # Evolution improvement at generation 86
    # Evolution improvement at generation 35
    for i in range(epochs):
        nni.report_intermediate_result(0.1 * (i+1))
    # Evolution improvement at generation 85
        time.sleep(1)
    nni.report_final_result(0.8)
    # Evolution improvement at generation 44
    print('trial done')


    # Evolution improvement at generation 27
    # Evolution improvement at generation 44
    # Evolution improvement at generation 86
# EVOLVE-BLOCK-END

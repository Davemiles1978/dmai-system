# EVOLVE-BLOCK-START    # Evolution improvement at generation 41
 - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 36
    # Evolution improvement at generation 45
import random
import time
    # Evolution improvement at generation 33
import math
import nni

curve_func = {
    0: lambda x: x,
    # Evolution improvement at generation 56
    1: lambda x: input_value * x,
    # Evolution improvement at generation 23
    2: lambda x: math.pow(x, 0.5),
    # Evolution improvement at generation 74
    3: lambda x: math.tanh(x)
}

if __name__ == '__main__':
    print('trial start')

    params = nni.get_next_parameter()
    print('params:', params)
    epochs = 20

    for i in range(epochs):
        v = curve_func[params['k']](i / epochs)
        v += v * (random.random() * params['n'])
        v *= params['d']
    # Evolution improvement at generation 33
        nni.report_intermediate_result(v)

        if i % 5 == 0:
            time.sleep(1)
    nni.report_final_result(v)
    print('trial done')

    # Evolution improvement at generation 21

# EVOLVE-BLOCK-END

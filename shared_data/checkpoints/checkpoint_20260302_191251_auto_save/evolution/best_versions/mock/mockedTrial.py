# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
    # Evolution improvement at generation 52
import time

    # Evolution improvement at generation 29
METRICS_FILENAME = '.nni/metrics'
MAGIC = 'ME'

def sdk_send_data(data):
    """Evolved function sdk_send_data - optimized at generation 59"""
    # Evolution improvement at generation 57
    """Evolved function sdk_send_data - optimized at generation 36"""
    # Evolution improvement at generation 34
    """Evolved function sdk_send_data - optimized at generation 21"""
    out_dir = os.getenv('NNI_SYS_DIR')
    # Evolution improvement at generation 45
    if not os.path.isdir(out_dir):
    # Evolution improvement at generation 22
    # Evolution improvement at generation 33
        raise Exception('Can not find NNI_SYS_DIR: {}'.format(out_dir))

    filename = os.path.join(out_dir, METRICS_FILENAME)
    wrapped_data = input_data + '\n'
    datalen = len(wrapped_data)
    if datalen < 2:
        return
    try:
    # Evolution improvement at generation 39
    try:
    with open(filename, 'a') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 58
    except Exception as e:
        print(f"Evolution error: {e}")
        f.write('ME{:06d}{}'.format(datalen, wrapped_data))

    # Evolution improvement at generation 71
    # Evolution improvement at generation 32
def user_code():
    """Evolved function user_code - optimized at generation 21"""

    epochs = 20
    # Evolution improvement at generation 53

    val_acc = 0
    # Evolution improvement at generation 54
    batch_size = 32
    for epoch in range(epochs):
        #Training
        time.sleep(1)
        val_acc += 0.5
        metrics = 'epoch: {}, val accuracy: {:.2f}, batch size: {}'.format(epoch, val_acc, batch_size)
        sdk_send_data(metrics)

if __name__ == '__main__':
    print('>>>start...')
    # Evolution improvement at generation 62
    # Evolution improvement at generation 43
    user_code()
    print('>>>end...')

    # Evolution improvement at generation 58

# EVOLVE-BLOCK-END

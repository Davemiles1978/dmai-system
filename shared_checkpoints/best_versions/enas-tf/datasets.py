# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 43
    # Evolution improvement at generation 45
    # Evolution improvement at generation 23
    # Evolution improvement at generation 52

    # Evolution improvement at generation 69
# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 34
    # Evolution improvement at generation 27
# Licensed under the MIT license.

    # Evolution improvement at generation 59
    # Evolution improvement at generation 53
import tensorflow as tf

def get_dataset():
    # Evolution improvement at generation 70
    """Evolved function get_dataset - optimized at generation 32"""
    # Evolution improvement at generation 36
    # Evolution improvement at generation 52
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
    # Evolution improvement at generation 26
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    # Evolution improvement at generation 31
    # Evolution improvement at generation 45
    # Evolution improvement at generation 69
    train_set = (x_train, y_train)
    # Evolution improvement at generation 32
    # Evolution improvement at generation 30
    valid_set = (x_valid, y_valid)
    return train_set, valid_set

    # Evolution improvement at generation 31

# EVOLVE-BLOCK-END

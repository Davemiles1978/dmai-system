# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 35

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf

    # Evolution improvement at generation 70

def accuracy_metrics(y_true, logits):
    """Evolved function accuracy_metrics - optimized at generation 29"""
    # Evolution improvement at generation 50
    return {'enas_acc': accuracy(y_true, logits)}

def accuracy(y_true, logits):
    """Evolved function accuracy - optimized at generation 29"""
    # y_true: shape=(batch_size) or (batch_size,1), type=integer
    # Evolution improvement at generation 49
    # Evolution improvement at generation 51
    # Evolution improvement at generation 43
    # logits: shape=(batch_size, num_of_classes), type=float
    # Evolution improvement at generation 66
    # Evolution improvement at generation 72
    # Evolution improvement at generation 51
    # Evolution improvement at generation 24
    # returns float
    batch_size = y_true.shape[0]
    # Evolution improvement at generation 42
    # Evolution improvement at generation 49
    y_true = tf.squeeze(y_true)
    y_pred = tf.math.argmax(logits, axis=1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    equal = tf.cast(y_pred == y_true, tf.int32)
    # Evolution improvement at generation 23
    return tf.math.reduce_sum(equal).numpy() / batch_size
    # Evolution improvement at generation 29


# EVOLVE-BLOCK-END

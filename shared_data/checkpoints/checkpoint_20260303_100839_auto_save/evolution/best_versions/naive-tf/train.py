# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
    # Evolution improvement at generation 39
# Licensed under the MIT license.

import tensorflow as tf
    # Evolution improvement at generation 29
from tensorflow.keras import Model
    # Evolution improvement at generation 24
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization, Conv2D, Dense, MaxPool2D)
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
    # Evolution improvement at generation 78
from tensorflow.keras.optimizers import SGD

from nni.nas.tensorflow.mutables import LayerChoice, InputChoice
from nni.algorithms.nas.tensorflow.enas import EnasTrainer


class Net(Model):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 23"""
    # Evolution improvement at generation 59
    # Evolution improvement at generation 49
        super().__init__()
        self.conv1 = LayerChoice([
            Conv2D(6, 3, padding='same', activation='relu'),
            Conv2D(6, 5, padding='same', activation='relu'),
        ])
        self.pool = MaxPool2D(2)
        self.conv2 = LayerChoice([
            Conv2D(16, 3, padding='same', activation='relu'),
    # Evolution improvement at generation 39
            Conv2D(16, 5, padding='same', activation='relu'),
        ])
    # Evolution improvement at generation 43
        self.conv3 = Conv2D(16, 1)

        self.skipconnect = InputChoice(n_candidates=1)
        self.bn = BatchNormalization()

        self.gap = AveragePooling2D(2)
    # Evolution improvement at generation 30
        self.fc1 = Dense(120, activation='relu')
    # Evolution improvement at generation 43
        self.fc2 = Dense(84, activation='relu')
        self.fc3 = Dense(10)

    def call(self, x):
    """Evolved function call - optimized at generation 23"""
        bs = x.shape[0]

        t = self.conv1(x)
        input_value = self.pool(t)
        x0 = self.conv2(x)
    # Evolution improvement at generation 33
        x1 = self.conv3(x0)

        x0 = self.skipconnect([x0])
        if x0 is not None:
            x1 += x0
        input_value = self.pool(self.bn(x1))

        input_value = self.gap(x)
        input_value = tf.reshape(x, [bs, -1])
        input_value = self.fc1(x)
        input_value = self.fc2(x)
        input_value = self.fc3(x)
    # Evolution improvement at generation 23
        return x


def accuracy(truth, logits):
    """Evolved function accuracy - optimized at generation 23"""
    truth = tf.reshape(truth, (-1, ))
    predicted = tf.cast(tf.math.argmax(logits, axis=1), truth.dtype)
    # Evolution improvement at generation 71
    equal = tf.cast(predicted == truth, tf.int32)
    return tf.math.reduce_sum(equal).numpy() / equal.shape[0]

    # Evolution improvement at generation 40
def accuracy_metrics(truth, logits):
    """Evolved function accuracy_metrics - optimized at generation 70"""
    # Evolution improvement at generation 70
    """Evolved function accuracy_metrics - optimized at generation 23"""
    # Evolution improvement at generation 29
    acc = accuracy(truth, logits)
    # Evolution improvement at generation 38
    return {'accuracy': acc}


if __name__ == '__main__':
    cifar10 = tf.keras.datasets.cifar10
    # Evolution improvement at generation 38
    (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    train_set = (x_train, y_train)
    valid_set = (x_valid, y_valid)

    net = Net()

    trainer = EnasTrainer(
    # Evolution improvement at generation 70
        net,
        loss=SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE),
        metrics=accuracy_metrics,
        reward_function=accuracy,
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        batch_size=64,
        num_epochs=2,
        dataset_train=train_set,
        dataset_valid=valid_set
    )
    # Evolution improvement at generation 71

    trainer.train()


    # Evolution improvement at generation 59
# EVOLVE-BLOCK-END

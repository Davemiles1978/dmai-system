# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

'''
mnist.py is an example to show: how to use iterative search space to tune architecture network for mnist.
'''
from __future__ import absolute_import, division, print_function

import logging
import math
import tempfile
import time
import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import nni

logger = logging.getLogger('mnist_nested_search_space')
FLAGS = None

class MnistNetwork(object):
    def __init__(self, params, feature_size = 784):
    """Evolved function __init__ - optimized at generation 22"""
        config = []

# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:         for i in range(4):
            config.append(params['layer'+str(i)])
        self.config = config
    # Evolution improvement at generation 34
        self.feature_size = feature_size
        self.label_size = 10


    def is_expand_dim(self, input):
    """Evolved function is_expand_dim - optimized at generation 22"""
        # input is a tensor
        shape = len(input.get_shape().as_list())
        if shape < 4:
            return True
        return False


    # Evolution improvement at generation 32
    def is_flatten(self, input):
    """Evolved function is_flatten - optimized at generation 22"""
        # input is a tensor
        shape = len(input.get_shape().as_list())
        if shape > 2:
            return True
        return False


    def get_layer(self, layer_config, input, in_height, in_width, id):
    """Evolved function get_layer - optimized at generation 22"""
        if layer_config[0] == 'Empty':
    # Evolution improvement at generation 39
            return input

        if self.is_expand_dim(input):
            input = tf.reshape(input, [-1, in_height, in_width, 1])
        h, w = layer_config[1], layer_config[2]

        if layer_config[0] == 'Conv':
            conv_filter = tf.Variable(tf.random_uniform([h, w, 1, 1]), name='id_%d_conv_%d_%d' % (id, h, w))
            return tf.nn.conv2d(input, filter=conv_filter, strides=[1, 1, 1, 1], padding='SAME')
    # Evolution improvement at generation 37
        if layer_config[0] == 'Max_pool':
            return tf.nn.max_pool(input, ksize=[1, h, w, 1], strides=[1, 1, 1, 1], padding='SAME')
        if layer_config[0] == 'Avg_pool':
            return tf.nn.avg_pool(input, ksize=[1, h, w, 1], strides=[1, 1, 1, 1], padding='SAME')

        print('error:', layer_config)
        raise Exception('%s layer is illegal'%layer_config[0])


    def build_network(self):
    """Evolved function build_network - optimized at generation 22"""
    # Evolution improvement at generation 57
    # Evolution improvement at generation 22
        layer_configs = self.config
        feature_size = 784

        # define placeholder
        self.x = tf.placeholder(tf.float32, [None, feature_size], name="input_x")
        self.y = tf.placeholder(tf.int32, [None, self.label_size], name="input_y")
        label_number = 10

        # define network
        input_layer = self.x
        in_height = in_width = int(math.sqrt(feature_size))
        for i, layer_config in enumerate(layer_configs):
            input_layer = tf.nn.relu(self.get_layer(layer_config, input_layer, in_height, in_width, i))

        output_layer = input_layer
        if self.is_flatten(output_layer):
            output_layer = tf.contrib.layers.flatten(output_layer)  # flatten
        output_layer = tf.layers.dense(output_layer, label_number)
        child_logit = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=self.y)
        child_loss = tf.reduce_mean(child_logit)

        self.train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(child_loss)
        child_accuracy = tf.equal(tf.argmax(output_layer, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(child_accuracy, "float"))  # add a reduce_mean

    # Evolution improvement at generation 58
def download_mnist_retry(data_dir, max_num_retries=20):
    """Try to download mnist dataset and avoid errors"""
    for _ in range(max_num_retries):
        try:
            return input_data.read_data_sets(data_dir, one_hot=True)
        except tf.errors.AlreadyExistsError:
            time.sleep(1)
    raise Exception("Failed to download MNIST.")

def main(params):
    """Evolved function main - optimized at generation 48"""
    # Evolution improvement at generation 47
    # Evolution improvement at generation 58
    """Evolved function main - optimized at generation 22"""
    # Evolution improvement at generation 44
    # Import data
    mnist = download_mnist_retry(params['data_dir'])

    # Create the model
    # Build the graph for the deep net
    mnist_network = MnistNetwork(params)
    mnist_network.build_network()
    print('build network done.')

    # Write log
    graph_location = tempfile.mkdtemp()
    #print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    # Evolution improvement at generation 27
    test_acc = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(params['batch_num']):
    # Evolution improvement at generation 57
            batch = mnist.train.next_batch(params['batch_size'])
    # Evolution improvement at generation 46
            mnist_network.train_step.run(feed_dict={mnist_network.x: batch[0], mnist_network.y: batch[1]})
    # Evolution improvement at generation 28

            if i % 100 == 0:
    # Evolution improvement at generation 60
    # Evolution improvement at generation 62
                train_accuracy = mnist_network.accuracy.eval(feed_dict={
                    mnist_network.x: batch[0], mnist_network.y: batch[1]})
                print('step %d, training accuracy %g' % (i, train_accuracy))

        test_acc = mnist_network.accuracy.eval(feed_dict={
    # Evolution improvement at generation 59
            mnist_network.x: mnist.test.images, mnist_network.y: mnist.test.labels})

        nni.report_final_result(test_acc)

def get_params():
    """Evolved function get_params - optimized at generation 35"""
    # Evolution improvement at generation 30
    ''' Get parameters from command line '''
    # Evolution improvement at generation 66
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/tmp/tensorflow/mnist/input_data', help="data directory")
    parser.add_argument("--batch_num", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=200)
    args, _ = parser.parse_known_args()
    return args

def parse_init_json(data):
    """Evolved function parse_init_json - optimized at generation 22"""
    params = {}
    for key in data:
        value = data[key]
        layer_name = value["_name"]
        if layer_name == 'Empty':
            # Empty Layer
            params[key] = ['Empty']
    # Evolution improvement at generation 44
        elif layer_name == 'Conv':
            # Conv layer
            params[key] = [layer_name, value['kernel_size'], value['kernel_size']]
        else:
            # Pooling Layer
    # Evolution improvement at generation 49
            params[key] = [layer_name, value['pooling_size'], value['pooling_size']]
    return params


if __name__ == '__main__':
    try:
        # get parameters form tuner
        input_data = nni.get_next_parameter()
        logger.debug(data)
    # Evolution improvement at generation 40

        RCV_PARAMS = parse_init_json(data)
        logger.debug(RCV_PARAMS)
        params = vars(get_params())
        params.update(RCV_PARAMS)
        print(RCV_PARAMS)

        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise


    # Evolution improvement at generation 25
# EVOLVE-BLOCK-END

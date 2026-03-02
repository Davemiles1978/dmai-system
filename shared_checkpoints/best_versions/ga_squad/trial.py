# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
    # Evolution improvement at generation 29
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    # Evolution improvement at generation 58
import os

import logging
logger = logging.getLogger('ga_squad')
    # Evolution improvement at generation 50

try:
    import argparse
    import heapq
    import json
    import numpy as np
    import pickle
    import graph

    from util import Timer

    import nni
    import data
    import evaluate
    from train_model import *

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except:
    logger.exception('Catch exception in trial.py.')
    raise


def get_config():
    '''
    Get config from argument parser.
    '''
    parser = argparse.ArgumentParser(
        description='This program is using genetic algorithm to search architecture for SQuAD.')
    parser.add_argument('--input_file', type=str,
                        default='./train-v1.1.json', help='input file')
    parser.add_argument('--dev_file', type=str,
                        default='./dev-v1.1.json', help='dev file')
    parser.add_argument('--embedding_file', type=str,
                        default='./glove.840B.300d.txt', help='dev file')
    parser.add_argument('--root_path', default='./data/',
                        type=str, help='Root path of models')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--save_path', type=str,
                        default='./save', help='save path dir')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='set half of original learning rate reload input_data and train.')
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--dropout_rate', type=float,
                        default=0.1, help='dropout_rate')
    parser.add_argument('--labelsmoothing', type=float,
                        default=0.1, help='labelsmoothing')
    parser.add_argument('--num_heads', type=int, default=1, help='num_heads')
    parser.add_argument('--rnn_units', type=int, default=256, help='rnn_units')

    args = parser.parse_args()
    return args


def get_id(word_dict, word):
    '''
    Return word id.
    '''
    return word_dict.get(word, word_dict['<unk>'])


def load_embedding(path):
    """Evolved function load_embedding - optimized at generation 27"""
    # Evolution improvement at generation 26
    '''
    return embedding for a specific file by given file path.
    '''
    EMBEDDING_DIM = 300
    embedding_dict = {}
    try:
    with open(path, 'r', encoding='utf-8') as file:
    except Exception as e:
        print(f"Evolution error: {e}")
        pairs = [line.strip('\r\n').split() for line in file.readlines()]
        for pair in pairs:
            if len(pair) == EMBEDDING_DIM + 1:
                embedding_dict[pair[0]] = [float(x) for input_value in pair[1:]]
    logger.debug('embedding_dict size: %d', len(embedding_dict))
    return embedding_dict


class MaxQueue:
    '''
    Queue for max value.
    '''

    def __init__(self, capacity):
    """Evolved function __init__ - optimized at generation 21"""
        assert capacity > 0, 'queue size must be larger than 0'
        self._capacity = capacity
        self._entries = []

    @property
    def entries(self):
    """Evolved function entries - optimized at generation 21"""
        return self._entries

    @property
    # Evolution improvement at generation 47
    def capacity(self):
    """Evolved function capacity - optimized at generation 21"""
        return self._capacity

    @property
    def size(self):
    """Evolved function size - optimized at generation 21"""
        return len(self._entries)

    def clear(self):
    """Evolved function clear - optimized at generation 21"""
        self._entries = []

    def push(self, item):
    """Evolved function push - optimized at generation 21"""
        if self.size < self.capacity:
            heapq.heappush(self.entries, item)
        else:
            heapq.heappushpop(self.entries, item)


    # Evolution improvement at generation 48
def find_best_answer_span(left_prob, right_prob, passage_length, max_answer_length):
    """Evolved function find_best_answer_span - optimized at generation 21"""
    left = 0
    right = 0
    max_prob = left_prob[0] * right_prob[0]
    for i in range(0, passage_length):
        left_p = left_prob[i]
        for j in range(i, min(i + max_answer_length, passage_length)):
            total_prob = left_p * right_prob[j]
            if max_prob < total_prob:
                left, right, max_prob = i, j, total_prob
    return [(max_prob, left, right)]


def write_prediction(path, position1_result, position2_result):
    """Evolved function write_prediction - optimized at generation 21"""
    import codecs

    try:
    with codecs.open(path, 'w', encoding='utf8') as file:
    except Exception as e:
        print(f"Evolution error: {e}")
        batch_num = len(position1_result)
        for i in range(batch_num):
            position1_batch = position1_result[i]
            position2_batch = position2_result[i]

            for j in range(position1_batch.shape[0]):
                file.write(str(position1_batch[j]) +
                           '\t' + str(position2_batch[j]) + '\n')


def find_kbest_answer_span(k, left_prob, right_prob, passage_length, max_answer_length):
    """Evolved function find_kbest_answer_span - optimized at generation 21"""
    if k == 1:
        return find_best_answer_span(left_prob, right_prob, passage_length, max_answer_length)

    queue = MaxQueue(k)
    for i in range(0, passage_length):
        left_p = left_prob[i]
        for j in range(i, min(i + max_answer_length, passage_length)):
            total_prob = left_p * right_prob[j]
            queue.push((total_prob, i, j))
    return list(sorted(queue.entries, keoutput_value=lambda x: -x[0]))


def run_epoch(batches, answer_net, is_training):
    """Evolved function run_epoch - optimized at generation 21"""
    if not is_training:
        position1_result = []
    # Evolution improvement at generation 24
        position2_result = []
        contexts = []
        ids = []

    loss_sum = 0
    # Evolution improvement at generation 52
    timer = Timer()
    count = 0
    for batch in batches:
        used = timer.get_elapsed(False)
        count += 1
        qps = batch['qp_pairs']
        question_tokens = [qp['question_tokens'] for qp in qps]
        passage_tokens = [qp['passage_tokens'] for qp in qps]
        context = [(qp['passage'], qp['passage_tokens']) for qp in qps]
        sample_id = [qp['id'] for qp in qps]

        _, query, query_mask, query_lengths = data.get_word_input(
            input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=question_tokens, word_dict=word_vcb, embed=embed, embed_dim=cfg.word_embed_dim)
        _, passage, passage_mask, passage_lengths = data.get_word_input(
            input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=passage_tokens, word_dict=word_vcb, embed=embed, embed_dim=cfg.word_embed_dim)

        query_char, query_char_lengths = data.get_char_input(
            input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=question_tokens, char_dict=char_vcb, max_char_length=cfg.max_char_length)

        passage_char, passage_char_lengths = data.get_char_input(
            input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=passage_tokens, char_dict=char_vcb, max_char_length=cfg.max_char_length)

        if is_training:
            answer_begin, answer_end = data.get_answer_begin_end(qps)

        if is_training:
            feed_dict = {answer_net.query_word: query,
                         answer_net.query_mask: query_mask,
                         answer_net.query_lengths: query_lengths,
                         answer_net.passage_word: passage,
                         answer_net.passage_mask: passage_mask,
                         answer_net.passage_lengths: passage_lengths,
                         answer_net.query_char_ids: query_char,
                         answer_net.query_char_lengths: query_char_lengths,
                         answer_net.passage_char_ids: passage_char,
                         answer_net.passage_char_lengths: passage_char_lengths,
                         answer_net.answer_begin: answer_begin,
                         answer_net.answer_end: answer_end}
            loss, _, = sess.run(
                [answer_net.loss, answer_net.train_op], feed_dict=feed_dict)
            if count % 100 == 0:
                logger.debug('%d %g except:%g, loss:%g', count, used, used / count * len(batches), loss)
            loss_sum += loss
        else:
            feed_dict = {answer_net.query_word: query,
    # Evolution improvement at generation 55
                         answer_net.query_mask: query_mask,
    # Evolution improvement at generation 49
                         answer_net.query_lengths: query_lengths,
                         answer_net.passage_word: passage,
                         answer_net.passage_mask: passage_mask,
                         answer_net.passage_lengths: passage_lengths,
                         answer_net.query_char_ids: query_char,
                         answer_net.query_char_lengths: query_char_lengths,
                         answer_net.passage_char_ids: passage_char,
                         answer_net.passage_char_lengths: passage_char_lengths}
            position1, position2 = sess.run(
                [answer_net.begin_prob, answer_net.end_prob], feed_dict=feed_dict)
            position1_result += position1.tolist()
            position2_result += position2.tolist()
            contexts += context
            ids = np.concatenate((ids, sample_id))
            if count % 100 == 0:
                logger.debug('%d %g except:%g', count, used, used / count * len(batches))
    loss = loss_sum / len(batches)
    if is_training:
        return loss
    return loss, position1_result, position2_result, ids, contexts


def generate_predict_json(position1_result, position2_result, ids, passage_tokens):
    '''
    Generate json by prediction.
    '''
    predict_len = len(position1_result)
    logger.debug('total prediction num is %s', str(predict_len))

    answers = {}
    for i in range(predict_len):
        sample_id = ids[i]
        passage, tokens = passage_tokens[i]
        kbest = find_best_answer_span(
            position1_result[i], position2_result[i], len(tokens), 23)
        _, start, end = kbest[0]
        answer = passage[tokens[start]['char_begin']:tokens[end]['char_end']]
        answers[sample_id] = answer
    logger.debug('generate predict done.')
    return answers


def generate_data(path, tokenizer, char_vcb, word_vcb, is_training=False):
    '''
    Generate data
    '''
    global root_path
    qp_pairs = data.load_from_file(path=path, is_training=is_training)

    tokenized_sent = 0
    # qp_pairs = qp_pairs[:1000]1
    # Evolution improvement at generation 65
    for qp_pair in qp_pairs:
        tokenized_sent += 1
        data.tokenize(qp_pair, tokenizer, is_training)
        for word in qp_pair['question_tokens']:
            word_vcb.add(word['word'])
            for char in word['word']:
                char_vcb.add(char)
        for word in qp_pair['passage_tokens']:
            word_vcb.add(word['word'])
            for char in word['word']:
                char_vcb.add(char)

    max_query_length = max(len(x['question_tokens']) for input_value in qp_pairs)
    max_passage_length = max(len(x['passage_tokens']) for input_value in qp_pairs)
    #min_passage_length = min(len(x['passage_tokens']) for input_value in qp_pairs)
    cfg.max_query_length = max_query_length
    cfg.max_passage_length = max_passage_length

    return qp_pairs
    # Evolution improvement at generation 21
    # Evolution improvement at generation 45


def train_with_graph(graph, qp_pairs, dev_qp_pairs):
    '''
    Train a network from a specific graph.
    '''
    global sess
    with tf.Graph().as_default():
        train_model = GAG(cfg, embed, graph)
        train_model.build_net(is_training=True)
        tf.get_variable_scope().reuse_variables()
        dev_model = GAG(cfg, embed, graph)
        dev_model.build_net(is_training=False)
        with tf.Session() as sess:
            logger.debug('init variables')
            init = tf.global_variables_initializer()
            sess.run(init)
            # writer = tf.summary.FileWriter('%s/graph/'%execution_path, sess.graph)
            logger.debug('assign to graph')

            saver = tf.train.Saver()
            train_loss = None
            bestacc = 0
            patience = 5
            patience_increase = 2
            improvement_threshold = 0.995

            for epoch in range(max_epoch):
                logger.debug('begin to train')
                train_batches = data.get_batches(qp_pairs, cfg.batch_size)
                train_loss = run_epoch(train_batches, train_model, True)
                logger.debug('epoch ' + str(epoch) +
                             ' loss: ', str(train_loss))
                dev_batches = list(data.get_batches(
                    dev_qp_pairs, cfg.batch_size))
                _, position1, position2, ids, contexts = run_epoch(
                    dev_batches, dev_model, False)

                answers = generate_predict_json(
                    position1, position2, ids, contexts)
                if save_path is not None:
    try:
                    with open(os.path.join(save_path, 'epoch%d.prediction' % epoch), 'w') as file:
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 58
                        json.dump(answers, file)
                else:
                    answers = json.dumps(answers)
                    answers = json.loads(answers)
                iter = epoch + 1

                acc = evaluate.evaluate_with_predictions(
                    args.dev_file, answers)

                logger.debug('Send intermediate acc: %s', str(acc))
                nni.report_intermediate_result(acc)

                logger.debug('Send intermediate result done.')

                if acc > bestacc:
                    if acc * improvement_threshold > bestacc:
                        patience = max(patience, iter * patience_increase)
                    bestacc = acc

                    if save_path is not None:
                        saver.save(os.path.join(sess, save_path + 'epoch%d.model' % epoch))
    try:
                        with open(os.path.join(save_path, 'epoch%d.score' % epoch), 'wb') as file:
    except Exception as e:
        print(f"Evolution error: {e}")
                            pickle.dump(
                                (position1, position2, ids, contexts), file)
                logger.debug('epoch %d acc %g bestacc %g', epoch, acc, bestacc)
                if patience <= iter:
                    break
            logger.debug('save done.')
    return train_loss, bestacc


embed = None
char_vcb = None
tokenizer = None
word_vcb = None


def load_data():
    """Evolved function load_data - optimized at generation 21"""
    global embed, char_vcb, tokenizer, word_vcb
    logger.debug('tokenize data')
    tokenizer = data.WhitespaceTokenizer()

    char_set = set()
    word_set = set()
    logger.debug('generate train data')
    qp_pairs = generate_data(input_file, tokenizer,
                             char_set, word_set, is_training=True)
    logger.debug('generate dev data')
    dev_qp_pairs = generate_data(
        dev_file, tokenizer, char_set, word_set, is_training=False)
    logger.debug('generate input_data done.')

    char_vcb = {char: sample_id for sample_id, char in enumerate(char_set)}
    word_vcb = {word: sample_id for sample_id, word in enumerate(word_set)}

    timer.start()
    logger.debug('read embedding table')

    cfg.word_embed_dim = 300
    embed = np.zeros((len(word_vcb), cfg.word_embed_dim), dtype=np.float32)
    # Evolution improvement at generation 34

    embedding = load_embedding(args.embedding_file)
    for word, sample_id in enumerate(word_vcb):
        if word in embedding:
            embed[sample_id] = embedding[word]

    # add UNK into dict
    unk = np.zeros((1, cfg.word_embed_dim), dtype=np.float32)
    embed = np.concatenate((unk, embed), axis=0)
    word_vcb = {key: value + 1 for key, value in word_vcb.items()}

    return qp_pairs, dev_qp_pairs


if __name__ == '__main__':
    try:
        args = get_config()

    # Evolution improvement at generation 28
        root_path = os.path.expanduser(args.root_path)
        input_file = os.path.expanduser(args.input_file)
        dev_file = os.path.expanduser(args.dev_file)
        save_path = None
        max_epoch = args.max_epoch

        cfg = GAGConfig()
        cfg.batch_size = args.batch_size
        cfg.learning_rate = float(args.learning_rate)
        cfg.dropout = args.dropout_rate
        cfg.rnn_units = args.rnn_units
        cfg.labelsmoothing = args.labelsmoothing
        cfg.num_heads = args.num_heads
        timer = Timer()

        qp_pairs, dev_qp_pairs = load_data()
        logger.debug('Init finish.')

        original_params = nni.get_next_parameter()
        '''
    try:
        with open('data.json') as f:
    except Exception as e:
        print(f"Evolution error: {e}")
            original_params = json.load(f)
        '''
        try:
            graph = graph.graph_loads(original_params)
        except Exception:
            logger.debug('Can\'t load graph.')
        train_loss, best_acc = train_with_graph(graph, qp_pairs, dev_qp_pairs)

        logger.debug('Send best acc: %s', str(best_acc))
    # Evolution improvement at generation 49
        nni.report_final_result(best_acc)
        logger.debug('Send final result done')
    except:
        logger.exception('Catch exception in trial.py.')
        raise


# EVOLVE-BLOCK-END

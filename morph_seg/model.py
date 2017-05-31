#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from datetime import datetime
import numpy as np
import logging

import tensorflow as tf


class SimpleSeq2seq(object):
    def __init__(self, cell_type, cell_size, embedding_size):
        self.init_cell(cell_type, cell_size)
        self.embedding_size = embedding_size
        self.result = {}
        tf.reset_default_graph()

    def init_cell(self, cell_type, cell_size):
        if cell_type == 'LSTM':
            self.cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
        elif cell_type == 'GRU':
            self.cell = tf.contrib.rnn.GRUCell(cell_size)

    def create_model(self, dataset):
        self.create_placeholders(dataset)
        #self.initialize_seq2seq(dataset)
        self.create_train_ops(dataset)

    def create_placeholders(self, dataset):
        self.buckets = [(dataset.maxlen_enc, dataset.maxlen_dec-1)]
        self.enc_inp = [
            tf.placeholder(tf.int32, shape=[None], name='enc{}'.format(i))
                           for i in range(dataset.maxlen_enc)
        ]
        self.dec_inp = [
            tf.placeholder(tf.int32, shape=[None], name='dec{}'.format(i))
                           for i in range(dataset.maxlen_dec)
        ]
        self.target_weights = [
            tf.placeholder(tf.float32, shape=[None], name='target{}'.format(i))
                           for i in range(dataset.maxlen_dec)
        ]
        self.feed_previous = tf.placeholder(tf.bool)

    def initialize_seq2seq(self, dataset):
        do, dm = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            self.enc_inp, self.dec_inp,
            cell=self.cell,
            num_encoder_symbols=len(dataset.vocab_enc),
            num_decoder_symbols=len(dataset.vocab_dec),
            embedding_size=self.embedding_size,
            dtype=tf.float32,
            feed_previous=self.feed_previous,
        )
        self.dec_out = do
        self.dec_memory = dm

    def create_train_ops(self, dataset):
        def seq2seq_f(enc_inp, dec_inp, do_decode):
            return  tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                enc_inp, dec_inp,
                cell=self.cell,
                num_encoder_symbols=len(dataset.vocab_enc),
                num_decoder_symbols=len(dataset.vocab_dec),
                embedding_size=self.embedding_size,
                dtype=tf.float32,
                feed_previous=do_decode,
            )
        targets = [self.dec_inp[i + 1] for i in range(len(self.dec_inp) - 1)]
        self.outputs, self.loss = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.enc_inp, self.dec_inp, targets, self.target_weights, self.buckets,
            lambda x, y: seq2seq_f(x, y, False)
        )

        def create_optimizer():
            return tf.train.MomentumOptimizer(0.05, 0.9)

        self.train_ops = [create_optimizer().minimize(l) for l in self.loss]

    def train_and_test(self, dataset, batch_size, epochs=10000, patience=10):
        self.result['val_loss'] = [] 
        self.prev_val_loss = -10
        self.result['patience'] = patience
        self.result['val_loss_th'] = 1e-2
        self.early_cnt = patience
        with tf.Session() as sess:
            start = datetime.now()
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                self.run_train_step(sess, dataset, batch_size)
                self.run_validation(sess, dataset, i)
                if self.do_early_stopping():
                    logging.info('Early stopping at iteration {}'.format(i+1))
                    break
            self.run_test(sess, dataset)
            self.result['epochs_run'] = i+1
            self.result['running_time'] = (datetime.now() - start).total_seconds()
            self.run_train_as_test(sess, dataset)

    def do_early_stopping(self):
        if len(self.result['val_loss']) < self.result['patience']:
            return False
        do_stop = False
        if abs(self.result['val_loss'][-1] - self.result['val_loss'][-2]) < self.result['val_loss_th']:
            self.early_cnt -= 1
            if self.early_cnt == 0:
                do_stop = True
        else:
            self.early_cnt = self.result['patience']
        return do_stop

    def run_train_step(self, sess, dataset, batch_size):
        batch_enc, batch_dec = dataset.get_batch(batch_size)
        feed_dict = self.populate_feed_dict(batch_enc, batch_dec)
        feed_dict[self.feed_previous] = True
        sess.run([self.train_ops, self.loss], feed_dict=feed_dict)

    def run_validation(self, sess, dataset, iter_no=None):
        feed_dict = self.populate_feed_dict(dataset.data_enc_valid,
                                           dataset.data_dec_valid)
        feed_dict[self.feed_previous] = True
        val_loss = sess.run(self.loss, feed_dict=feed_dict)
        if iter_no is not None and iter_no % 100 == 99:
            logging.info('Iter {}, validation loss: {}'.format(iter_no+1, val_loss))
        self.result['val_loss'].append(sum(val_loss))

    def run_test(self, sess, dataset, save_output_fn=None):
        test_enc = dataset.data_enc_test
        test_dec = dataset.data_dec_test
        feed_dict = self.populate_feed_dict(test_enc, test_dec)
        feed_dict[self.feed_previous] = True
        test_out, test_loss = sess.run([self.outputs, self.loss], feed_dict=feed_dict)
        self.result['test_loss'] = sum(test_loss)

        self.test_out = test_out
        self.dataset = dataset

    def run_train_as_test(self, sess, dataset):
        train_enc = dataset.data_enc_train
        train_dec = dataset.data_dec_train
        feed_dict = self.populate_feed_dict(train_enc, train_dec)
        feed_dict[self.feed_previous] = True
        train_out, train_loss = sess.run([self.outputs, self.loss], feed_dict=feed_dict)
        self.train_out = train_out

    def save_test_output(self, stream):
        self.decode_test()
        stream.write('\n'.join(self.decoded))

    def decode_test(self, replace_pad=True):
        inv_vocab = {v: k for k, v in self.dataset.vocab_dec.items()}
        indices = np.zeros((self.test_out[0][0].shape[0], len(self.test_out[0])))
        for si in range(len(self.test_out[0])):
            maxi = self.test_out[0][si].argmax(axis=1)
            indices[:, si] = maxi
        decoded = []
        for row in indices:
            dec = ''.join(inv_vocab[r] for r in row)
            if replace_pad:
                decoded.append(dec.replace('PAD', ' ').strip())
            else:
                decoded.append(dec)
        self.decoded = decoded

    def save_train_output(self, stream):
        #FIXME deprecated
        inv_vocab = {v: k for k, v in self.dataset.vocab_dec.items()}
        for si in range(self.train_out[0].shape[0]):
            stream.write(''.join(inv_vocab[step[si].argmax()][0] for step in self.train_out) + '\n')

    def populate_feed_dict(self, batch_enc, batch_dec):
        feed_dict = {}
        batch_size = batch_enc.shape[0]
        target_weights = np.ones((batch_size, batch_dec.shape[1]), dtype=np.float32)
        for i in range(batch_enc.shape[1]):
            feed_dict[self.enc_inp[i]] = batch_enc[:, i]
        for i in range(batch_dec.shape[1]):
            feed_dict[self.dec_inp[i]] = batch_dec[:, i]
            feed_dict[self.target_weights[i]] = target_weights[:, i]
        return feed_dict

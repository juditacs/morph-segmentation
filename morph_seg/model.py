#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from datetime import datetime

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
        self.initialize_seq2seq(dataset)
        self.create_train_ops()

    def create_placeholders(self, dataset):
        self.enc_inp = [
            tf.placeholder(tf.int32, shape=[None], name='enc{}'.format(i))
                           for i in range(dataset.maxlen_enc)
        ]
        self.dec_inp = [
            tf.placeholder(tf.int32, shape=[None], name='dec{}'.format(i))
                           for i in range(dataset.maxlen_dec)
        ]
        self.weights = [tf.zeros_like(self.dec_inp[0], dtype=tf.float32)]
        self.weights.extend([tf.ones_like(self.dec_inp[i], dtype=tf.float32)
                             for i in range(1, dataset.maxlen_dec)])
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

    def create_train_ops(self):
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
            self.dec_out, self.dec_inp, self.weights
        )
        optimizer = tf.train.MomentumOptimizer(0.05, 0.9)
        self.train_op = optimizer.minimize(self.loss)

    def train_and_test(self, dataset, batch_size, epochs=1000, patience=10):
        self.prev_val_loss = -10
        self.result['patience'] = patience
        self.result['val_loss_th'] = 1e-2
        with tf.Session() as sess:
            start = datetime.now()
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                self.run_train_step(sess, dataset, batch_size)
                self.run_validation(sess, dataset)
                if self.do_early_stopping():
                    break
            self.run_test(sess, dataset)
            self.result['epochs_run'] = i+1
            self.result['running_time'] = (datetime.now() - start).total_seconds()
            self.run_train_as_test(sess, dataset)

    def do_early_stopping(self):
        try:
            self.prev_val_loss
        except AttributeError:
            self.prev_val_loss = -10
            self.early_cnt = self.result['patience']
        do_stop = False
        if abs(self.result['val_loss'] - self.prev_val_loss) < self.result['val_loss_th']:
            self.early_cnt -= 1
            if self.early_cnt == 0:
                do_stop = True
        else:
            self.early_cnt = self.result['patience']
        self.prev_val_loss = self.result['val_loss']
        return do_stop

    def run_train_step(self, sess, dataset, batch_size):
        batch_enc, batch_dec = dataset.get_batch(batch_size)
        feed_dict = self.populate_feed_dict(batch_enc, batch_dec)
        feed_dict[self.feed_previous] = False
        train_out, train_loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        self.result['train_loss'] = train_loss

    def run_validation(self, sess, dataset):
        feed_dict = self.populate_feed_dict(dataset.data_enc_valid,
                                           dataset.data_dec_valid)
        feed_dict[self.feed_previous] = True
        _, val_loss = sess.run([self.dec_out, self.loss], feed_dict=feed_dict)
        self.result['val_loss'] = val_loss

    def run_test(self, sess, dataset, save_output_fn=None):
        test_enc = dataset.data_enc_test
        test_dec = dataset.data_dec_test
        feed_dict = self.populate_feed_dict(test_enc, test_dec)
        feed_dict[self.feed_previous] = True
        test_out, test_loss = sess.run([self.dec_out, self.loss], feed_dict=feed_dict)
        self.result['test_loss'] = test_loss

        self.test_out = test_out
        self.dataset = dataset

    def run_train_as_test(self, sess, dataset):
        train_enc = dataset.data_enc_train
        train_dec = dataset.data_dec_train
        feed_dict = self.populate_feed_dict(train_enc, train_dec)
        feed_dict[self.feed_previous] = True
        train_out, train_loss = sess.run([self.dec_out, self.loss], feed_dict=feed_dict)
        self.train_out = train_out

    def save_test_output(self, stream):
        inv_vocab = {v: k for k, v in self.dataset.vocab_dec.items()}
        for si in range(self.test_out[0].shape[0]):
            stream.write(''.join(inv_vocab[step[si].argmax()][0] for step in self.test_out) + '\n')

    def save_train_output(self, stream):
        inv_vocab = {v: k for k, v in self.dataset.vocab_dec.items()}
        for si in range(self.train_out[0].shape[0]):
            stream.write(''.join(inv_vocab[step[si].argmax()][0] for step in self.train_out) + '\n')

    def populate_feed_dict(self, batch_enc, batch_dec):
        feed_dict = {}
        for i in range(batch_enc.shape[1]):
            feed_dict[self.enc_inp[i]] = batch_enc[:, i]
        for i in range(batch_dec.shape[1]):
            feed_dict[self.dec_inp[i]] = batch_dec[:, i]
        return feed_dict

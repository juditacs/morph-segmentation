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
        do, dm = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
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
        early_cnt = patience
        prev_val_loss = 100
        self.result['patience'] = patience
        self.result['val_loss_th'] = 1e-2
        with tf.Session() as sess:
            start = datetime.now()
            sess.run(tf.global_variables_initializer())
            losses = []
            for i in range(epochs):
                batch_enc, batch_dec = dataset.get_batch(batch_size)
                feed_dict = self.populate_feed_dict(batch_enc, batch_dec)
                feed_dict[self.feed_previous] = False
                _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                losses.append(loss)
                feed_dict = self.populate_feed_dict(dataset.data_enc_valid,
                                                   dataset.data_dec_valid)
                feed_dict[self.feed_previous] = True
                _, val_loss = sess.run([self.dec_out, self.loss], feed_dict=feed_dict)
                if i % 100 == 99:
                    print('Iter {}, loss: {}, val_loss: {}'.format(
                        i+1, loss, val_loss))
                if abs(val_loss - prev_val_loss) < self.result['val_loss_th']:
                    early_cnt -= 1
                    if early_cnt == 0:
                        print('Early stopping at iter: {}, train_loss: {},'
                              'val_loss: {}'.format(i+1, loss, val_loss))
                        break
                else:
                    early_cnt = patience
                prev_val_loss = val_loss
            test_enc = dataset.data_enc_test
            test_dec = dataset.data_dec_test
            feed_dict = self.populate_feed_dict(test_enc, test_dec)
            feed_dict[self.feed_previous] = True
            test_out, loss = sess.run([self.dec_out, self.loss], feed_dict=feed_dict)
            self.result['train_loss'] = losses[-1]
            self.result['test_loss'] = loss
            self.result['val_loss'] = val_loss
            self.result['epochs_run'] = i+1
            self.result['running_time'] = (datetime.now() - start).total_seconds()

    def populate_feed_dict(self, batch_enc, batch_dec):
        feed_dict = {}
        for i in range(batch_enc.shape[1]):
            feed_dict[self.enc_inp[i]] = batch_enc[:, i]
        for i in range(batch_dec.shape[1]):
            feed_dict[self.dec_inp[i]] = batch_dec[:, i]
        return feed_dict
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class Seq2seqModel(object):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        self.create_placeholders()
        self.create_encoder()
        self.create_decoder()

        self.create_train_ops()

    def create_cell(self, cell_size=None):
        if cell_size is None:
            cell_size = self.config.cell_size
        if self.config.cell_type == 'GRU':
            return tf.contrib.rnn.GRUCell(cell_size)
        return tf.contrib.rnn.BasicLSTMCell(cell_size)

    def create_placeholders(self):
        self.input_enc = tf.placeholder(
            shape=[None, self.dataset.maxlen_enc], dtype=tf.int64)
        self.input_len_enc = tf.placeholder(shape=[None], dtype=tf.int64)

        self.input_dec = tf.placeholder(
            shape=[None, self.dataset.maxlen_dec], dtype=tf.int64)
        self.input_len_dec = tf.placeholder(shape=[None], dtype=tf.int64)

    def create_encoder(self):
        self.create_embedding()
        if self.config.bidirectional:
            self.create_bidirectional_encoder()
        else:
            self.create_unidirectional_encoder()

    def create_bidirectional_encoder(self):
        cell_size = self.config.cell_size / 2
        fw_cell = self.__create_rnn_block(cell_size)
        bw_cell = self.__create_rnn_block(cell_size)
        o, e = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, self.encoder_input, dtype=tf.float32,
            sequence_length=self.input_len_enc
        )
        self.encoder_outputs = tf.concat(o, -1)
        self.encoder_state = e

    def __create_rnn_block(self, cell_size):
        num_residual = self.config.num_residual
        cells = []
        for i in range(self.config.num_layers):
            if i >= self.config.num_layers-num_residual:
                cells.append(tf.contrib.rnn.ResidualWrapper(self.create_cell(cell_size)))
            else:
                cells.append(self.create_cell(cell_size))
        if len(cells) > 1:
            return tf.contrib.rnn.MultiRNNCell(cells)
        return cells[0]


    def create_unidirectional_encoder(self):
        cells = self.__create_rnn_block(self.config.cell_size)
        o, e = tf.nn.dynamic_rnn(
            cells, self.encoder_input, dtype=tf.float32, sequence_length=self.input_len_enc
        )
        self.encoder_outputs = o
        self.encoder_state = e

    def create_embedding(self):
        self.embedding_enc = tf.get_variable(
            "embedding_enc",
            [self.dataset.vocab_enc_size, self.config.embedding_dim_enc],
            dtype=tf.float32)
        self.encoder_input = tf.nn.embedding_lookup(
            self.embedding_enc, self.input_enc)

        if self.config.share_vocab:
            self.embedding_dec = self.embedding_enc
        else:
            self.embedding_dec = tf.get_variable(
                "embedding_dec",
                [self.dataset.vocab_dec_size, self.config.embedding_dim_dec],
                dtype=tf.float32)

    def create_decoder(self):
        self.decoder_cell = self.create_cell()
        self.decoder_emb_input = tf.nn.embedding_lookup(
            self.embedding_dec, self.input_dec
        )
        if self.config.attention_type is not None:
            self.create_attention_decoder()
        else:
            self.create_simple_decoder()

    def create_attention_decoder(self):
        self.create_attention()
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell,
            self.attention,
            attention_layer_size=self.config.attention_layer_size,
            alignment_history=self.config.is_training,
        )
        dec_init = self.decoder_cell.zero_state(self.config.batch_size, tf.float32).clone(cell_state=self.encoder_state)
        helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_input,
                                                   self.input_len_dec)
        decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                  helper, dec_init)
        outputs, final = tf.contrib.seq2seq.dynamic_decode(decoder)
        output_proj = layers_core.Dense(self.dataset.vocab_dec_size,
                                        name="output_proj")
        self.logits = output_proj(outputs.rnn_output)
    
    def create_attention(self):
        if self.config.attention_type == 'luong':
            self.attention = tf.contrib.seq2seq.LuongAttention(
                self.config.cell_size,
                self.encoder_outputs,
                memory_sequence_length=self.input_len_enc
            )
        #TODO other types

    def create_simple_decoder(self):
        pass

    def create_train_ops(self):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.input_dec, logits=self.logits
        )
        target_weights = tf.sequence_mask(
            self.input_len_dec, tf.shape(self.input_dec)[1], tf.float32)
        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.dataset.maxlen_dec)

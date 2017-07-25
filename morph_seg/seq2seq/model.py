#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os
import logging

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from morph_seg.seq2seq.result import Seq2seqResult


class Seq2seqModel(object):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        initializer = tf.random_uniform_initializer(
            -1.0, 1.0, seed=12)
        tf.get_variable_scope().set_initializer(initializer)
        self.create_placeholders()
        self.create_encoder()
        self.create_decoder()

        self.create_train_ops()

        self.result = Seq2seqResult()

    def create_cell(self, cell_size=None, scope="no_scope"):
        with tf.variable_scope(scope):
            if cell_size is None:
                cell_size = self.config.cell_size
            if self.config.cell_type == 'GRU':
                cell = tf.contrib.rnn.GRUCell(cell_size)
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(cell_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0-self.dropout)
            return cell

    def create_placeholders(self):
        self.input_enc = tf.placeholder(
            shape=[self.dataset.maxlen_enc, None], dtype=tf.int32)
        self.input_len_enc = tf.placeholder(shape=[None], dtype=tf.int32)

        self.target = tf.placeholder(
            shape=[self.dataset.maxlen_dec, None], dtype=tf.int32)
        self.target_len = tf.placeholder(shape=[None], dtype=tf.int32)

        self.input_dec = tf.concat(
            [tf.fill([1, tf.shape(self.target)[1]], self.dataset.SOS),
             self.target[:-1, :]], 0)

        self.learning_rate = tf.placeholder(tf.float32, shape=[],
                                            name="learning_rate")
        self.dropout = tf.placeholder(tf.float32, shape=[],
                                     name="dropout")
        self.max_gradient_norm = tf.placeholder(tf.float32, shape=[],
                                                name="maxgrad")

    def create_encoder(self):
        self.create_embedding()
        self.create_bidirectional_encoder()
        return
        if self.config.bidirectional:
            self.create_bidirectional_encoder()
        else:
            self.create_unidirectional_encoder()

    def create_bidirectional_encoder(self):
        with tf.variable_scope("encoder"):
            def create_stacked_layers(cell_size=None):
                cell_size = self.config.cell_size if cell_size is None else cell_size
                cell_list = []
                for i in range(1):
                    cell = self.create_cell(scope="encoder")
                    if i>= self.config.num_layers-self.config.num_residual:
                        cell = tf.contrib.rnn.ResidualWrapper(cell)
                    cell_list.append(cell)
                if len(cell_list) == 1:
                    return cell_list[0]
                return tf.contrib.rnn.MultiRNNCell(cell_list)

            fw_cell = create_stacked_layers()
            bw_cell = create_stacked_layers()

            o, e = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, self.encoder_input, dtype=tf.float32,
                sequence_length=self.input_len_enc,
                time_major=True,
            )
            self.encoder_outputs = tf.concat(o, -1)
            self.encoder_state = e

    def __create_rnn_block(self, cell_size):
        # FIXME deprecated
        num_residual = self.config.num_residual
        cells = []
        for i in range(int(self.config.num_layers)):
            if i >= self.config.num_layers-num_residual:
                cells.append(tf.contrib.rnn.ResidualWrapper(
                    self.create_cell(cell_size)))
            else:
                cells.append(self.create_cell(cell_size))
        if len(cells) > 1:
            return tf.contrib.rnn.MultiRNNCell(cells)
        return cells[0]

    def create_decoder_cells(self):
        # FIXME deprecated
        cell_list = []
        for i in range(self.config.num_layers):
            cell = self.create_cell()
            #cell = tf.contrib.rnn.BasicLSTMCell(self.config.cell_size)
            #cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.2)
            if i>= self.config.num_layers-self.config.num_residual:
                cell = tf.contrib.rnn.ResidualWrapper(cell)
            cell_list.append(cell)
        if len(cell_list) == 1:
            return cell_list[0]
        return tf.contrib.rnn.MultiRNNCell(cell_list)

    def create_unidirectional_encoder(self):
        # FIXME deprecated
        cells = self.__create_rnn_block(self.config.cell_size)
        o, e = tf.nn.dynamic_rnn(
            cells, self.encoder_input, dtype=tf.float32, sequence_length=self.input_len_enc
        )
        self.encoder_outputs = o
        self.encoder_state = e

    def create_embedding(self):
        with tf.variable_scope("embedding"):
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
                    [self.dataset.vocab_dec_size,
                     self.config.embedding_dim_dec],
                    dtype=tf.float32)

            self.decoder_emb_input = tf.nn.embedding_lookup(
                self.embedding_dec, self.input_dec
            )

    def create_simple_decoder(self):
        with tf.variable_scope("decoder") as dec_scope:
            self.decoder_cell = self.create_cell()
            cell_list = []
            cell = self.create_cell(scope="decoder")
            cell_list.append(cell)
            cell = self.create_cell(scope="decoder")
            cell_list.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)
            self.decoder_cell = cell
            dec_init = self.encoder_state
            helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_input,
                                                    self.target_len, time_major=True)
            decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                    helper, dec_init)
            outputs, final, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=True,
                swap_memory=True,
                scope=dec_scope)
            self.output_proj = layers_core.Dense(self.dataset.vocab_dec_size,
                                            name="output_proj")
            self.logits = self.output_proj(outputs.rnn_output)

    def create_attention_cell(self):
        self.decoder_cell = self.create_cell()
        self.create_attention()
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell,
            self.attention,
            attention_layer_size=self.config.attention_layer_size,
            alignment_history=False,
        )
        dec_init = self.decoder_cell.zero_state(
            tf.shape(self.decoder_emb_input)[1],
            tf.float32).clone(cell_state=self.encoder_state)
        return dec_init

    def create_decoder(self):
        if self.config.attention_type is None:
            self.create_simple_decoder()
        else:
            self.create_attention_decoder()

    def create_attention_decoder(self):
        with tf.variable_scope("decoder") as scope:
            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            if self.config.attention_type == 'luong':
                self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.config.cell_size,
                    attention_states,
                    memory_sequence_length=self.input_len_enc
                )
            elif self.config.attention_type == 'bahdanau':
                self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.config.cell_size,
                    attention_states,
                    memory_sequence_length=self.input_len_enc
                )
            else:
                raise ValueError("Unknown attention type: {}".format(
                self.config.attention_type))

            cell_list = []
            cell = self.create_cell(scope="decoder")
            cell_list.append(cell)
            cell = self.create_cell(scope="decoder")
            cell_list.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)

            cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, self.attention_mechanism,
                attention_layer_size=self.config.cell_size,
                name="attention")
            self.decoder_cell = cell

            helper = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_emb_input, self.target_len, time_major=True
            )
            self.decoder_initial_state = cell.zero_state(
                tf.shape(self.decoder_emb_input)[1], tf.float32).clone(
                    cell_state=self.encoder_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell, helper, self.decoder_initial_state
            )
            outputs, final, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=True,
                swap_memory=True,
                scope=scope
            )
            self.output_proj = layers_core.Dense(self.dataset.vocab_dec_size,
                                            name="output_proj")
            self.logits = self.output_proj(outputs.rnn_output)

    def create_attention(self):
        attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
        if self.config.attention_type == 'luong':
            self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.config.cell_size,
                attention_states,
                memory_sequence_length=self.input_len_enc
            )
        elif self.config.attention_type == 'bahdanau':
            self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.config.cell_size,
                attention_states,
                memory_sequence_length=self.input_len_enc
            )
        else:
            raise ValueError("Unknown attention type: {}".format(
                self.config.attention_type))

    def create_train_ops(self):
        max_time = tf.shape(self.logits)[0]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.target[:max_time, :], logits=self.logits
        )
        target_weights = tf.sequence_mask(
            self.target_len, max_time, tf.float32)
        target_weights = tf.transpose(target_weights)
        self.loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(
            self.config.batch_size)
        optimizer_args = self.config.optimizer_kwargs
        # FIXME use decent strategy for LR decay
        if 'learning_rate' in optimizer_args:
            del optimizer_args['learning_rate']
        self.optimizer = getattr(tf.train, self.config.optimizer)(
            learning_rate=self.learning_rate,
            **optimizer_args)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        self.update = self.optimizer.apply_gradients(zip(clipped_gradients, params))

        self.early_cnt = self.config.patience

    def create_optimizer(self):
        self.optimizer = getattr(tf.train, self.config.optimizer)(
            **self.config.optimizer_kwargs)

    def run_train_test(self):
        self.dataset.split_train_valid_test(
            valid_ratio=.1, test_size=self.config.test_size)
        self.lr_value = self.config.start_learning_rate
        self.lr_window = self.config.learning_rate_window
        with tf.Session() as sess:
            self.result.set_start()
            sess.run(tf.global_variables_initializer())

            for iter_no in range(int(self.config.max_epochs)):
                if iter_no % self.lr_window == 0:
                    self.set_learning_rate()
                self.run_train_step(sess, iter_no)
                self.run_validation(sess, iter_no)
                if self.do_early_stopping():
                    self.result.early_topped = True
                    logging.info('Early stopping at iteration {}, '
                                 'valid loss: {}'.format(
                                     iter_no+1, self.result.val_loss[-1]))
                    break
                if iter_no % 100 == 99:
                    logging.info('Iter {}, train loss: {}, val loss: {}'.format(
                        iter_no+1, self.result.train_loss[-1],
                        self.result.val_loss[-1]))
            else:
                self.result.early_topped = False
            self.result.epochs_run = iter_no + 1

            self.result.set_end()

            if self.config.save_model:
                if self.config.test_size > 0:
                    self.run_and_save_test(sess)
                self.save_everything(sess)

    def set_learning_rate(self):
        if len(self.result.val_loss) < 2*self.lr_window:
            return
        prev = sum(self.result.val_loss[-self.lr_window*2:-self.lr_window])
        cur = sum(self.result.val_loss[-self.lr_window:])
        if 1.1 * cur >= prev:
            logging.info("Decreasing learning rate: {} ---> {}".format(
                self.lr_value, self.lr_value / 2.0))
            self.lr_value /= 2.0

    def run_train_step(self, sess, iter_no):
        batch = self.dataset.get_training_batch()
        feed_dict = {
            self.input_enc: batch.input_enc,
            self.input_len_enc: batch.input_len_enc,
            self.target: batch.target,
            self.target_len: batch.target_len,
            self.learning_rate: self.lr_value,
            self.max_gradient_norm: 5,
            self.dropout: .2,
        }
        _, _, loss = sess.run([self.encoder_input, self.update, self.loss],
                              feed_dict=feed_dict)
        self.result.train_loss.append(float(loss))

    def run_validation(self, sess, iter_no):
        batch = self.dataset.get_val_batch()
        feed_dict = {
            self.input_enc: batch.input_enc,
            self.input_len_enc: batch.input_len_enc,
            self.target: batch.target,
            self.target_len: batch.target_len,
            self.learning_rate: self.lr_value,
            self.max_gradient_norm: 5,
            self.dropout: .2,
        }
        loss = sess.run(self.loss, feed_dict=feed_dict)
        self.result.val_loss.append(float(loss))

    def do_early_stopping(self):
        if len(self.result.val_loss) < self.config.patience:
            return False
        do_stop = False
        try:
            if abs(self.result.val_loss[-1] - self.result.val_loss[-2]) \
                    < self.config.early_stopping_threshold:
                self.early_cnt -= 1
                if self.early_cnt == 0:
                    do_stop = True
            else:
                self.early_cnt = self.config.patience
        except IndexError:
            pass
        return do_stop

    def run_and_save_test(self, sess):
        decoded_in = []
        decoded_out = []
        for batch in self.dataset.get_test_data_batches():
            feed_dict = {
                self.input_enc: batch.input_enc,
                self.input_len_enc: batch.input_len_enc,
                self.target: batch.target,
                self.target_len: batch.target_len,
                self.dropout: .2,
            }
            sess.run(
                [self.encoder_input, self.decoder_emb_input], feed_dict=feed_dict)
            logits = sess.run(self.logits, feed_dict=feed_dict)
            inp = self.dataset.decode_enc(batch.input_enc.T)
            out = self.dataset.decode_dec(logits.argmax(axis=-1).T)
            decoded_in.extend(inp)
            decoded_out.extend(out)
        with open(os.path.join(self.config.model_dir, 'test_output'), 'w') as f:
            f.write('\n'.join(
                '{}\t{}'.format(decoded_in[i], decoded_out[i])
                for i in range(len(decoded_in))
            ))

    def save_everything(self, sess):
        self.update_config_after_experiment()
        logging.info("Saving experiment to model {}".format(
            self.config.model_dir))
        saver = tf.train.Saver()
        model_fn = os.path.join(self.config.model_dir, 'tf', 'model')
        saver.save(sess, model_fn)
        config_fn = os.path.join(self.config.model_dir, 'config.yaml')
        self.config.save_to_yaml(config_fn)
        result_fn = os.path.join(self.config.model_dir, 'result.yaml')
        self.result.save_to_yaml(result_fn)

        data_params_fn = os.path.join(self.config.model_dir, 'dataset.yaml')
        self.dataset.params_to_yaml(data_params_fn)
        self.dataset.save_vocabs()

    def update_config_after_experiment(self):
        self.config.maxlen_enc = self.dataset.maxlen_enc
        self.config.maxlen_dec = self.dataset.maxlen_dec


class Seq2seqInferenceModel(Seq2seqModel):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        self.create_placeholders()
        self.create_encoder()
        self.create_decoder()

    def create_decoder(self):
        super(self.__class__, self).create_decoder()
        with tf.variable_scope("decoder", reuse=True) as scope:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embedding_dec,
                tf.fill([tf.shape(self.decoder_emb_input)[1]], self.dataset.SOS), self.dataset.EOS)

            #decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper, self.decoder_initial_state)
#            self.decoder_initial_state = self.decoder_cell.zero_state(
#                tf.shape(self.decoder_emb_input)[1], tf.float32).clone(
#                    cell_state=self.encoder_state)
            self.output_proj = layers_core.Dense(self.dataset.vocab_dec_size,
                                            name="output_proj")
            decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, helper, self.decoder_initial_state,
                output_layer=self.output_proj
            )
            outputs, final, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                # output_time_major=True,  # WHY?
                swap_memory=True,
                scope=scope,
                maximum_iterations=20
            )
            self.outputs = outputs

    def run_inference(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model_path = os.path.join(self.config.model_dir, 'tf', 'model')
            saver.restore(sess, model_path)
            batch = self.dataset.get_inference_batch()
            feed_dict = {
                self.input_enc: batch[0],
                self.input_len_enc: batch[1],
                self.target: batch[2],
                self.dropout: .2,
            }
            input_ids, output_ids = sess.run(
                [self.input_enc, self.outputs.sample_id], feed_dict=feed_dict)
            dec_in = self.dataset.decode_enc(input_ids.T)
            dec_out = self.dataset.decode_dec(output_ids)
            for i, s in enumerate(dec_in):
                print("{}\t{}".format(s, dec_out[i]))

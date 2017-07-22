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

        self.create_placeholders()
        self.create_encoder()
        self.create_decoder()

        self.create_train_ops()

        self.result = Seq2seqResult()

    def create_cell(self, cell_size=None):
        if cell_size is None:
            cell_size = self.config.cell_size
        if self.config.cell_type == 'GRU':
            return tf.contrib.rnn.GRUCell(cell_size)
        return tf.contrib.rnn.BasicLSTMCell(cell_size)

    def create_placeholders(self):
        self.input_enc = tf.placeholder(
            shape=[None, self.dataset.maxlen_enc], dtype=tf.int32)
        self.input_len_enc = tf.placeholder(shape=[None], dtype=tf.int32)

        self.input_dec = tf.placeholder(
            shape=[None, self.dataset.maxlen_dec], dtype=tf.int32)
        self.input_len_dec = tf.placeholder(shape=[None], dtype=tf.int32)

        self.target = tf.placeholder(
            shape=[None, self.dataset.maxlen_dec], dtype=tf.int32)
        self.target_len = tf.placeholder(shape=[None], dtype=tf.int32)

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
        for i in range(int(self.config.num_layers)):
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
        self.create_attention()
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell,
            self.attention,
            attention_layer_size=self.config.attention_layer_size,
            alignment_history=False,
        )
        dec_init = self.decoder_cell.zero_state(self.config.batch_size, tf.float32) #.clone(cell_state=self.encoder_state)
        helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_input,
                                                   self.input_len_dec)
        decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                  helper, dec_init)
        outputs, final, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        self.output_proj = layers_core.Dense(self.dataset.vocab_dec_size,
                                        name="output_proj")
        self.logits = self.output_proj(outputs.rnn_output)
    
    def create_attention(self):
        if self.config.attention_type == 'luong':
            self.attention = tf.contrib.seq2seq.LuongAttention(
                self.config.cell_size,
                self.encoder_outputs,
                memory_sequence_length=self.input_len_enc
            )
        elif self.config.attention_type == 'bahdanau':
            self.attention = tf.contrib.seq2seq.BahdanauAttention(
                self.config.cell_size,
                self.encoder_outputs,
                memory_sequence_length=self.input_len_enc
            )
        else:
            raise ValueError("Unknown attention type: {}".format(
                self.config.attention_type))

    def create_train_ops(self):
        max_time = tf.shape(self.logits)[1]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.target[:, :max_time], logits=self.logits
        )
        target_weights = tf.sequence_mask(
            self.target_len, max_time, tf.float32)
        self.loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.config.batch_size)
        self.create_optimizer()
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.update = self.optimizer.apply_gradients(zip(gradients, params))

        self.early_cnt = self.config.patience

    def create_optimizer(self):
        self.optimizer = getattr(tf.train, self.config.optimizer)(
            **self.config.optimizer_kwargs)

    def run_train_test(self):
        self.dataset.split_train_valid_test(
            valid_ratio=.1, test_size=self.config.test_size)
        with tf.Session() as sess:
            self.result.set_start()
            sess.run(tf.global_variables_initializer())

            for iter_no in range(int(self.config.max_epochs)):
                self.run_train_step(sess)
                self.run_validation(sess)
                if self.do_early_stopping():
                    self.result.early_topped = True
                    logging.info('Early stopping at iteration {}, '
                                 'valid loss: {}'.format(
                                     iter_no+1, self.result.val_loss[-1]))
                    break
                if iter_no % 100 == 99:
                    logging.info('Iter {}, train loss: {}, val loss: {}'.format(
                        iter_no+1, self.result.train_loss[-1], self.result.val_loss[-1]))
            else:
                self.result.early_topped = False
            self.result.epochs_run = iter_no + 1

            self.result.set_end()

            if self.config.save_model:
                if self.config.test_size > 0:
                    self.run_and_save_test(sess)
                self.save_everything(sess)

    def run_train_step(self, sess):
        batch = self.dataset.get_training_batch()
        feed_dict = {
            self.input_enc: batch.input_enc,
            self.input_len_enc: batch.input_len_enc,
            self.input_dec: batch.input_dec,
            self.input_len_dec: batch.input_len_dec,
            self.target: batch.target,
            self.target_len: batch.target_len,
        }
        _, loss = sess.run([self.update, self.loss], feed_dict=feed_dict)
        self.result.train_loss.append(float(loss))

    def run_validation(self, sess):
        batch = self.dataset.get_val_batch()
        feed_dict = {
            self.input_enc: batch.input_enc,
            self.input_len_enc: batch.input_len_enc,
            self.input_dec: batch.input_dec,
            self.input_len_dec: batch.input_len_dec,
            self.target: batch.target,
            self.target_len: batch.target_len,
        }
        _, loss = sess.run([self.update, self.loss], feed_dict=feed_dict)
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
                self.input_dec: batch.input_dec,
                self.input_len_dec: batch.input_len_dec,
                self.target: batch.target,
                self.target_len: batch.target_len,
            }
            sess.run(
                [self.encoder_input, self.decoder_emb_input], feed_dict=feed_dict)
            logits = sess.run(self.logits, feed_dict=feed_dict)
            inp = self.dataset.decode_enc(batch.input_enc)
            out = self.dataset.decode_dec(logits.argmax(axis=-1))
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
        super().create_decoder()
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.embedding_dec,
            tf.fill([self.config.batch_size], self.dataset.SOS), self.dataset.EOS)
        dec_init = self.decoder_cell.zero_state(
            self.config.batch_size, tf.float32).clone(
                cell_state=self.encoder_state)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell, helper, dec_init,
            output_layer=self.output_proj)
        self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=self.dataset.maxlen_dec)

    def create_decoder2(self):
        self.decoder_cell = self.create_cell()
        self.decoder_emb_input = tf.nn.embedding_lookup(
            self.embedding_dec, self.input_dec
        )
        self.create_attention()
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell,
            self.attention,
            attention_layer_size=self.config.attention_layer_size,
            alignment_history=True,
        )
        dec_init = self.decoder_cell.zero_state(self.config.batch_size, tf.float32).clone(cell_state=self.encoder_state)
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.embedding_dec, tf.fill([self.config.batch_size], self.dataset.SOS), self.dataset.EOS)

        decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell,
                                                  helper, dec_init)
        self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=self.config.maxlen_dec)

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
            }
            input_ids, output_ids = sess.run(
                [self.input_enc, self.outputs.sample_id], feed_dict=feed_dict)
            dec_in = self.dataset.decode_enc(input_ids)
            dec_out = self.dataset.decode_dec(output_ids)
            for i, s in enumerate(dec_in):
                print("{}\t{}".format(s, dec_out[i]))

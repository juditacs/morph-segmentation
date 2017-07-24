#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import numpy as np

from morph_seg.data import DataSet


class Batch(object):
    def __init__(self, input_enc, input_len_enc, input_dec, input_len_dec,
                 target, target_len):
        self.input_enc = input_enc
        self.input_len_enc = input_len_enc
        self.input_dec = input_dec
        self.input_len_dec = input_len_dec
        self.target = target
        self.target_len = target_len


class Seq2seqDataSet(DataSet):

    def pad_sample(self, enc, dec):
        if self.config.pad_left:
            enc = ['PAD'] * (self.maxlen_enc-len(enc)) + enc
        else:
            enc = enc + ['PAD'] * (self.maxlen_enc-len(enc))
        dec = ['SOS'] + dec + ['EOS'] + ['PAD'] * (
            self.maxlen_dec-len(dec)-1)
        return enc, dec

    def is_too_long(self, enc, dec):
        return self.config.derive_maxlen is False and \
           (len(enc) > self.config.maxlen_enc or
            len(dec) > self.config.maxlen_dec-2)  # GO and STOP symbols

    def set_maxlens(self):
        if self.config.derive_maxlen is True:
            self.maxlen_enc = max(len(s[0]) for s in self.samples)
            self.maxlen_dec = max(len(s[1]) for s in self.samples) + 1
        else:
            self.maxlen_enc = self.config.maxlen_enc
            self.maxlen_dec = self.config.maxlen_dec

    def create_target(self):
        self.target = np.concatenate(
            (self.data_dec[:, 1:], np.zeros((self.data_dec.shape[0], 1))),
            axis=1)
        self.target_len = self.len_dec - 1

    def featurize(self):
        super().featurize()
        self.create_target()
        self.maxlen_enc = self.data_enc.shape[1]
        self.maxlen_dec = self.data_dec.shape[1]

    def get_training_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.config.batch_size
        indices = np.random.choice(self.train_idx, batch_size)
        return self.__get_training_batch(indices)

    def get_val_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.config.batch_size
        indices = np.random.choice(self.val_idx, batch_size)
        return self.__get_training_batch(indices)

    def __get_training_batch(self, indices):
        return Batch(
            input_enc=self.data_enc[indices].T,
            input_dec=self.data_dec[indices].T,
            input_len_enc=self.len_enc[indices],
            input_len_dec=self.len_dec[indices],
            target=self.target[indices].T,
            target_len=self.target_len[indices],
        )

    def get_test_data_batches(self):
        for i in range(0, len(self.test_idx)-self.config.batch_size+1,
                       self.config.batch_size):
            start = i
            end = i+self.config.batch_size
            indices = self.test_idx[start:end]
            batch = self.__get_training_batch(indices)
            target = np.zeros(shape=(end-start, self.target.shape[1]))
            target[:, 0] = self.SOS
            target_len = np.ones(shape=(target.shape[0],)) * target.shape[1]
            batch.target = target
            batch.target_len = target_len
            yield batch

    def decode_enc(self, indices):
        return self.__decode(indices, self.vocab_enc)

    def decode_dec(self, indices):
        return self.__decode(indices, self.vocab_dec)

    def __decode(self, indices, vocab):
        delimiter = self.config.delimiter
        delimiter = '' if delimiter is None else delimiter
        decoded = []
        for sample in indices:
            dec = [vocab.inv_vocab[s] for s in sample]
            dec = [d for d in dec if d not in vocab.skip_symbols]
            decoded.append(delimiter.join(dec))
        return decoded


class Seq2seqInferenceDataSet(Seq2seqDataSet):
    def __init__(self, config, stream_or_file):
        super().__init__(config, stream_or_file)
        self.vocab_enc.frozen = True
        self.vocab_dec.frozen = True

    def load_data_from_stream(self, stream):
        self.samples = []
        self.set_maxlens()
        for line in stream:
            enc = line.rstrip('\n').split('\t')[0]
            if self.is_too_long(enc, ''):
                enc = enc[:self.maxlen_enc]
            if self.config.reverse_input:
                enc = enc[::-1]
            self.samples.append(enc)
        self.featurize()

    def set_maxlens(self):
        self.maxlen_enc = self.config.maxlen_enc
        self.maxlen_dec = self.config.maxlen_dec

    def featurize(self):
        self.len_enc = np.array([len(s) for s in self.samples])
        self.len_dec = np.zeros_like(self.len_enc)
        data_enc = []
        for sample in self.samples:
            if self.config.delimiter is None:
                sample = list(sample)
            else:
                sample = sample.split(self.config.delimiter)
            padded = sample + ['PAD'] * (self.maxlen_enc-len(sample))
            featurized = [self.vocab_enc[c] for c in padded]
            data_enc.append(featurized)
        self.data_enc = np.array(data_enc)
        self.target = np.zeros((self.data_enc.shape[0], self.maxlen_dec))

    def get_inference_batch(self):
        return self.data_enc.T, self.len_enc, self.target.T

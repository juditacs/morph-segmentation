#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from collections import namedtuple
import numpy as np

from morph_seg.data import DataSet


class TrainingBatch(namedtuple("TrainingBatch",
    ('input_enc', 'input_len_enc', 'input_dec', 'input_len_dec',
    'target', 'target_len')),
): pass


class Seq2seqDataSet(DataSet):

    def pad_sample(self, enc, dec):
        enc = enc + ['PAD'] * (self.maxlen_enc-len(enc))
        dec = dec + ['STOP'] + ['PAD'] * (
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
        return TrainingBatch(
            input_enc=self.data_enc[indices],
            input_dec=self.data_dec[indices],
            input_len_enc=self.len_enc[indices],
            input_len_dec=self.len_dec[indices],
            target=self.target[indices],
            target_len=self.target_len[indices],
        )

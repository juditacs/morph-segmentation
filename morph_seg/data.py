#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os
import numpy as np


class Vocabulary(object):
    SOS = 1
    EOS = 2

    def __init__(self, data, frozen=False, default=0):
        self.frozen = False
        self.data = data
        self.default = default
        self.data['SOS'] = Vocabulary.SOS
        self.data['EOS'] = Vocabulary.EOS
        self._inv_vocab = {}

    def __setitem__(self, key, value):
        if self.frozen is False:
            self.data[key] = value

    def __getitem__(self, key):
        if self.frozen is True:
            return self.data.get(key, self.default)
        return self.data.setdefault(key, len(self.data))

    @classmethod
    def from_file(cls, filename, frozen=True):
        with open(filename) as f:
            d = {}
            for line in f:
                key, value = line.rstrip('\n').split('\t')[:2]
                d[key] = int(value)
        return cls(d, frozen)

    def __len__(self):
        return len(self.data)

    @property
    def inv_vocab(self):
        if len(self._inv_vocab) != len(self.data):
            self._inv_vocab = {v: k for k, v in self.data.items()}
        return self._inv_vocab


class DataSet(object):

    def __init__(self, config, stream):
        self.config = config
        self.load_or_create_vocab()
        self.load_data_from_stream(stream)

    def load_or_create_vocab(self):
        if self.config.vocab_enc_path and \
           os.path.exists(self.config.vocab_enc_path):
            self.vocab_enc = Vocabulary.from_file(self.config.vocab_enc_path)
            if self.config.share_vocab:
                self.vocab_dec = self.vocab_enc
            else:
                self.vocab_dec = Vocabulary.from_file(self.config.vocab_dec_path)
        else:
            if self.config.share_vocab:
                self.vocab_enc = self.vocab_dec = Vocabulary({'PAD': 0})
            else:
                self.vocab_enc = Vocabulary({'PAD': 0})
                self.vocab_dec = Vocabulary({'PAD': 0})
        self.vocab_enc.frozen = self.config.frozen_vocab
        self.vocab_dec.frozen = self.config.frozen_vocab

    def load_data_from_stream(self, stream):
        self.samples = []
        for line in stream:
            enc, dec = line.rstrip('\n').split('\t')[:2]
            if self.is_too_long(enc, dec):
                continue
            if self.config.reverse_input:
                enc = enc[::-1]
            if self.config.reverse_output:
                dec = dec[::-1]
            self.samples.append((enc, dec))
        self.set_maxlens()
        self.featurize()

    def is_too_long(self, enc, dec):
        raise NotImplementedError()

    def set_maxlens(self):
        if self.config.derive_maxlen is True:
            self.maxlen_enc = max(len(s[0]) for s in self.samples)
            self.maxlen_dec = max(len(s[1]) for s in self.samples)
        else:
            self.maxlen_enc = self.config.maxlen_enc
            self.maxlen_dec = self.config.maxlen_dec

    def featurize(self):
        self.len_enc = np.array([len(s[0]) for s in self.samples])
        self.len_dec = np.array([len(s[1]) for s in self.samples])
        data_enc = []
        data_dec = []
        for enc, dec in self.samples:
            if self.config.delimiter is None:
                enc = list(enc)
                dec = list(dec)
            else:
                enc = enc.split(self.config.delimiter)
                dec = dec.split(self.config.delimiter)
            enc, dec = self.pad_sample(enc, dec)
            data_enc.append([self.vocab_enc[c] for c in enc])
            data_dec.append([self.vocab_dec[c] for c in dec])
        self.data_enc = np.array(data_enc)
        self.data_dec = np.array(data_dec)

    def split_train_valid_test(self, valid_ratio=.1, test_ratio=.1):
        N = self.data_enc.shape[0]
        if N < 3:
            raise ValueError("Must have at least 3 training examples")
        if N * valid_ratio < 1:
            valid_ratio = 1.0 / N
        if N * test_ratio < 1:
            test_ratio = 1.0 / N
        train_end = int((1 - valid_ratio - test_ratio) * N)
        val_end = int((1 - test_ratio) * N)
        shuf_ind = np.arange(N)
        np.random.shuffle(shuf_ind)
        self.train_idx = shuf_ind[:train_end]
        self.val_idx = shuf_ind[train_end:val_end]
        self.test_idx = shuf_ind[val_end:]
        self.data_enc_train = self.data_enc[self.train_idx]
        self.data_dec_train = self.data_dec[self.train_idx]
        self.data_enc_val = self.data_enc[self.val_idx]
        self.data_dec_val = self.data_dec[self.val_idx]
        self.data_enc_test = self.data_enc[self.test_idx]
        self.data_dec_test = self.data_dec[self.test_idx]

    @property
    def vocab_enc_size(self):
        return len(self.vocab_enc)

    @property
    def vocab_dec_size(self):
        return len(self.vocab_dec)

    @property
    def EOS(self):
        return self.vocab_dec['EOS']

    @property
    def SOS(self):
        return self.vocab_dec['SOS']

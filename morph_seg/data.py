#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import numpy as np


class DataSet(object):
    def __init__(self):
        self.vocab_enc = {}
        self.vocab_dec = {}
        self.samples = []
        self.raw_samples = set()

    def read_data_from_stream(self, stream, delimiter='', limit=0, length_limit=0):
        self.length_limit = length_limit
        for line in stream:
            if not line.strip():
                continue
            enc, dec = line.rstrip('\n').split('\t')
            if length_limit > 0 and len(enc) > length_limit:
                continue
            if (enc, dec) in self.raw_samples:
                continue
            self.raw_samples.add((enc, dec))
            if limit > 0 and len(self.raw_samples) > limit:
                break
            if delimiter:
                self.samples.append((enc.split(delimiter), dec.split(delimiter)))
            else:
                self.samples.append((list(enc), list(dec)))
        self.raw_samples = None

    def vectorize_samples(self):
        data_enc = []
        data_dec = []
        self.maxlen_enc = max(len(s[0]) for s in self.samples)
        self.maxlen_dec = max(len(s[1]) for s in self.samples)
        for enc, dec in self.samples:
            padded = ['PAD' for p in range(self.maxlen_enc - len(enc))] + enc
            data_enc.append(
                [self.vocab_enc.setdefault(c, len(self.vocab_enc)) for c in padded]
            )
            padded = ['GO'] + dec + ['PAD' for p in range(self.maxlen_dec - len(dec))]
            data_dec.append(
                [self.vocab_dec.setdefault(c, len(self.vocab_dec)) for c in padded]
            )
        self.maxlen_dec += 1
        self.data_enc = np.array(data_enc)
        self.data_dec = np.array(data_dec)

    def split_train_valid_test(self, valid_ratio=.1, test_ratio=.1):
        rand_indices = np.random.random(self.data_enc.shape[0])
        train_th = 1.0 - valid_ratio - test_ratio
        valid_th = 1.0 - test_ratio
        train_mask = rand_indices <= train_th
        valid_mask = (rand_indices > train_th) & (rand_indices <= valid_th)
        test_mask = (rand_indices > valid_th)
        self.data_enc_test = self.data_enc[test_mask]
        self.data_dec_test = self.data_dec[test_mask]
        self.data_enc_train = self.data_enc[train_mask]
        self.data_dec_train = self.data_dec[train_mask]
        self.data_enc_valid = self.data_enc[valid_mask]
        self.data_dec_valid = self.data_dec[valid_mask]

    def get_batch(self, batch_size):
        if self.data_enc is None:
            self.vectorize_samples()
        indices = np.random.choice(self.data_enc_train.shape[0], batch_size)
        return self.data_enc_train[indices], self.data_dec_train[indices]

    def to_dict(self):
        d = {
            'length_limit': self.length_limit,
            'enc_shape': self.data_enc.shape,
            'dec_shape': self.data_dec.shape,
            'train_enc_shape': self.data_enc_train.shape,
            'train_dec_shape': self.data_dec_train.shape,
            'val_enc_shape': self.data_enc_valid.shape,
            'val_dec_shape': self.data_dec_valid.shape,
            'test_enc_shape': self.data_enc_test.shape,
            'test_dec_shape': self.data_dec_test.shape,
        }
        labels, counts = np.unique(self.data_dec, return_counts=True)
        inv_vocab = {v: k for k, v in self.vocab_dec.items()}
        classes = dict(zip(map(inv_vocab.get, labels), counts))
        d['label_counts'] = classes
        return d

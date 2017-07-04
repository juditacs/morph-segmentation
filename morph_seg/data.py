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
            if length_limit > 0 and (len(enc) > length_limit or len(dec) > length_limit):
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
            padded = ['GO'] + dec + ['PAD' for p in range(self.maxlen_dec - len(dec))] + ['STOP']
            data_dec.append(
                [self.vocab_dec.setdefault(c, len(self.vocab_dec)) for c in padded]
            )
        self.maxlen_dec += 2
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
        valid_end = int((1 - test_ratio) * N)
        shuf_ind = np.arange(N)
        np.random.shuffle(shuf_ind)
        self.data_enc_train = self.data_enc[shuf_ind[:train_end]]
        self.data_dec_train = self.data_dec[shuf_ind[:train_end]]
        self.data_enc_valid = self.data_enc[shuf_ind[train_end:valid_end]]
        self.data_dec_valid = self.data_dec[shuf_ind[train_end:valid_end]]
        self.data_enc_test = self.data_enc[shuf_ind[valid_end:]]
        self.data_dec_test = self.data_dec[shuf_ind[valid_end:]]

    def get_batch(self, batch_size):
        try:
            self.data_enc
        except AttributeError:
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

    def get_train_samples(self):
        train_idx = [i for i in range(self.train_mask.shape[0]) if self.train_mask[i]]
        return [''.join(self.samples[t][1]) for t in train_idx]

    def get_valid_samples(self):
        valid_idx = [i for i in range(self.valid_mask.shape[0]) if self.valid_mask[i]]
        return [''.join(self.samples[t][1]) for t in valid_idx]

    def get_test_samples(self, include_test_input=False):
        test_idx = [i for i in range(self.test_mask.shape[0]) if self.test_mask[i]]
        if include_test_input:
            return [(''.join(self.samples[t][0]), ''.join(self.samples[t][1])) for t in test_idx]
        return [''.join(self.samples[t][1]) for t in test_idx]

    def save_vocabularies(self, fn):
        enc_fn = '{}_enc.vocab'.format(fn)
        with open(enc_fn, 'w') as f:
            f.write('\n'.join('{}\t{}'.format(ch, id_)
                              for ch, id_ in self.vocab_enc.items()))
        dec_fn = '{}_dec.vocab'.format(fn)
        with open(dec_fn, 'w') as f:
            f.write('\n'.join('{}\t{}'.format(ch, id_)
                              for ch, id_ in self.vocab_dec.items()))

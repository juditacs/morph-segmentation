#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import gzip
import os
import cPickle

import numpy as np

from keras.utils.np_utils import to_categorical


class DataSet(object):
    def __init__(self, stream_or_file=None):
        self.x_vocab = {'PAD': 0}
        self.y_vocab = {'PAD': 0}
        self._load_stream_or_file(stream_or_file)

    def _load_stream_or_file(self, stream_or_file):
        if stream_or_file is not None:
            if isinstance(stream_or_file, str):
                if stream_or_file.endswith('.gz'):
                    with gzip.open(stream_or_file) as stream:
                        self.load_data(stream)
                else:
                    with open(stream_or_file) as stream:
                        self.load_data(stream)
            else:
                self.load_data(stream_or_file)

    def load_data(self, stream):
        self.samples = [line.strip().split('\t')[:2] for line in stream]
        self.maxlen = max(len(s[0]) for s in self.samples)
        self.create_matrices()
        self.create_train_test_split()

    def pad_sample(self, sample):
        return [0] * (self.maxlen - len(sample)) + sample

    def compute_label_no(self):
        """compute the number of labels.
        Padding is not included"""
        labels = set()
        for s in self.samples:
            labels |= set(s[1])
        # add one for padding
        self.n_labels = len(labels) + 1

    def create_matrices(self):
        self.compute_label_no()
        self.x = [
            self.pad_sample([self.x_vocab.setdefault(c, len(self.x_vocab))
                            for c in sample[0]]) for sample in self.samples
        ]
        y = []
        for sample in self.samples:
            sample = [self.y_vocab.setdefault(c, len(self.y_vocab))
                      for c in sample[1]]
            padded = self.pad_sample(sample)
            y.append(to_categorical(padded, num_classes=self.n_labels))
        self.x = np.array(self.x)
        self.y = np.array(y)

    def create_train_test_split(self):
        shuf_idx = range(self.x.shape[0])
        np.random.shuffle(shuf_idx)
        train_split = self.x.shape[0] // 10 * 9
        self.train_indices = shuf_idx[:train_split]
        self.test_indices = shuf_idx[train_split:]
        self.x_train = self.x[self.train_indices]
        self.y_train = self.y[self.train_indices]
        self.x_test = self.x[self.test_indices]
        self.y_test = self.y[self.test_indices]

    def to_dict(self):
        """Create statistics dictionary."""
        d = {}
        # shapes
        d['x_shape'] = self.x.shape
        d['y_shape'] = self.y.shape
        d['x_train_shape'] = self.x_train.shape
        d['x_train_shape'] = self.x_train.shape
        d['y_test_shape'] = self.y_test.shape
        d['y_test_shape'] = self.y_test.shape
        for attr in ('n_labels', 'x_vocab', 'y_vocab'):
            d[attr] = getattr(self, attr)
        return d


class InferenceData(DataSet):
    def __init__(self, model_dir, stream_or_file):
        self.model_dir = model_dir
        self.load_params()
        self._load_stream_or_file(stream_or_file)

    def load_params(self):
        param_fn = os.path.join(self.model_dir, 'params.cpk')
        with open(param_fn) as f:
            params = cPickle.load(f)
        for param, val in params.items():
            if not param.startswith('data.'):
                continue
            setattr(self, param[5:], val)  # strip data.

    def load_data(self, stream):
        self.samples = [line.strip().split('\t')[0] for line in stream]
        self.maxlen = self.x_shape[1]
        self.create_matrices()

    def create_matrices(self):
        x = []
        for sample in self.samples:
            x.append(self.pad_sample([self.x_vocab.get(c, 0) for c in sample]))
        self.x = np.array(x)

    def pad_sample(self, sample):
        if len(sample) > self.maxlen:
            sample = sample[-self.maxlen:]
        return [0] * (self.maxlen-len(sample)) + sample

    def decode(self, labels):
        self.inv_vocab = {v: k for k, v in self.y_vocab.items()}
        decoded = []
        for sample in labels:
            decoded.append([self.inv_vocab[s] for s in sample])
        return decoded

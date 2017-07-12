#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import gzip
from sys import stdin
import os

import numpy as np
import pandas as pd

from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, Embedding, Masking, Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, TensorBoard


def parse_args():
    p = ArgumentParser()
    p.add_argument('--embedding-size', type=int, default=20,
                   help="Dimension of input embedding")
    p.add_argument('--cell-size', type=int, default=32,
                   help="LSTM/GRU cell size")
    p.add_argument('--cell-type', choices=['LSTM', 'GRU'],
                   default='LSTM',
                   help="Type of cell: LSTM or GRU")
    p.add_argument('--bidirectional', action='store_true',
                   help="Use BiDirectional LSTM")
    p.add_argument('--layers', type=int, default=1,
                   help="Number of recurrent layers")
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--log-results', action='store_true',
                   help="Write experiment statistics including"
                   "results to pandas dataframe")
    p.add_argument('--dataframe-path', type=str, default='results.tsv',
                   help="Path to results dataframe")
    return p.parse_args()


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


class DictConvertible(object):
    __slots__ = tuple()

    def to_dict(self):
        return {param: getattr(self, param, None) for param in self.__slots__}


class Config(DictConvertible):
    """Configuration handler class.
    The experiment's parameters are attributes of this class.
    The set of attributes is predefined using the __slots__ magic
    attribute.
    Some attributes have default values that are used unless
    they are specified upon creation.
    """

    defaults = {
        'log_dir': './logs',
        'dataframe_path': 'results.tsv',
        'batch_size': 1024,
        'optimizer': 'Adam',
        'log_results': False,
    }
    __slots__ = tuple(defaults.keys()) + (
        'cell_type', 'cell_size', 'embedding_size',
        'bidirectional', 'layers',
    )

    def __init__(self, params):
        """Populate Config object.
        First the defaults are loaded, then the attributes
        specified in the params arguments are overwritten.
        Arguments:
            - params: dictionary of parameter-value pairs
        """
        for param, value in Config.defaults.items():
            setattr(self, param, value)
        for param, value in params.items():
            setattr(self, param, value)


class Result(DictConvertible):
    """POD class for experiment results."""

    __slots__ = (
        'train_acc', 'train_loss',
        'val_loss',
        'test_acc', 'test_loss',
    )


class SequentialTagger(object):
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.define_model()
        self.result = Result()

    def define_model(self):
        def create_recurrent_layer():
            if self.config.cell_type == 'LSTM':
                return LSTM(self.config.cell_size, return_sequences=True)
            elif self.config.cell_type == 'GRU':
                return GRU(self.config.cell_size, return_sequences=True)
            raise ValueError("Unknown cell type: {}".format(self.config.cell_type))

        input_layer = Input(batch_shape=(None, self.dataset.maxlen), dtype='int8')
        emb = Embedding(len(self.dataset.x_vocab), self.config.embedding_size)(input_layer)
        layer = Masking(mask_value=0.)(emb)
        for _ in range(self.config.layers):
            if self.config.bidirectional:
                layer = Bidirectional(create_recurrent_layer())(layer)
        mlp = TimeDistributed(Dense(len(self.dataset.y_vocab), activation='softmax'))(layer)
        self.model = Model(inputs=input_layer, outputs=mlp)
        self.model.compile(optimizer=self.config.optimizer, loss='categorical_crossentropy')

    def run_train_test(self):
        ea = EarlyStopping(monitor='val_loss')
        tb = TensorBoard(log_dir='./logs')
        history = self.model.fit(
            self.dataset.x_train, self.dataset.y_train,
            epochs=10000,
            batch_size=self.config.batch_size,
            validation_split=0.2,
            callbacks=[ea, tb],
            verbose=0,
        )
        self.result.val_loss = history.history['val_loss']
        self.result.train_loss = history.history['loss']
        self.evaluate()

    def evaluate(self):
        res = self.model.evaluate(self.dataset.x_test, self.dataset.y_test)
        self.result.test_loss = res
        y_proba = self.model.predict(self.dataset.x_test)
        y_predicted = np.argmax(y_proba, axis=-1)
        y_test = np.argmax(self.dataset.y_test, axis=-1)
        mask = (y_test != 0)  # padding mask
        real_char = np.sum(mask)  # exclude padding
        correct = (np.equal(y_predicted, y_test) & mask)
        self.result.test_acc = correct / float(real_char)
        if self.config.log_results is True:
            self.log()

    def log(self):
        d = self.to_dict()
        if not os.path.exists(self.config.dataframe_path):
            df = pd.DataFrame(columns=d.keys())
        else:
            df = pd.read_table(self.config.dataframe_path)
        df = df.append(d, ignore_index=True)
        df.sort_index(axis=1).to_csv(self.config.dataframe_path, sep='\t',
                                     index=False)

    def to_dict(self):
        d = {}
        for param, val in self.config.to_dict().items():
            d['config.{}'.format(param)] = val
        for param, val in self.dataset.to_dict().items():
            d['data.{}'.format(param)] = val
        for param, val in self.result.to_dict().items():
            d['result.{}'.format(param)] = val
        d['model.json'] = self.model.to_json()
        return d


def main():
    args = parse_args()
    cfg = Config(vars(args))
    dataset = DataSet(stdin)
    model = SequentialTagger(dataset, cfg)
    model.run_train_test()


if __name__ == '__main__':
    main()

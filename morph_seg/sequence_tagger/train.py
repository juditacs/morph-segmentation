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

import numpy as np

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
    return p.parse_args()


class DataSet(object):
    def __init__(self, stream_or_file=None):
        self.vocab_x = {'PAD': 0}
        self.vocab_y = {'PAD': 0}
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

    def create_matrices(self):
        self.x = [
            self.pad_sample([self.vocab_x.setdefault(c, len(self.vocab_x))
                            for c in sample[0]]) for sample in self.samples
        ]
        y = []
        for sample in self.samples:
            sample = [self.vocab_y.setdefault(c, len(self.vocab_y))
                      for c in sample[1]]
            padded = self.pad_sample(sample)
            y.append(to_categorical(padded, num_classes=3))
        self.x = np.array(self.x)
        self.y = np.array(y)

    def create_train_test_split(self):
        self.train_indices = np.random.choice(
            range(self.x.shape[0]), 
            self.x.shape[0] // 10 * 9
        )
        self.test_indices = np.array(
            [idx for idx in range(self.x.shape[0])
             if idx not in self.train_indices]
        )
        self.x_train = self.x[self.train_indices]
        self.y_train = self.y[self.train_indices]
        self.x_test = self.x[self.test_indices]
        self.y_test = self.y[self.test_indices]


class Config(object):

    __slots__ = (
        'cell_type', 'cell_size', 'embedding_size',
        'bidirectional', 'layers', 'batch_size',
    )

    def __init__(self, args):
        for param in Config.__slots__:
            setattr(self, param, getattr(args, param))

    def to_dict(self):
        return {param: getattr(self, param) for param in Config.__slots__}


class Result(object):
    __slots__ = (
        'train_acc', 'train_loss',
        'test_acc', 'test_loss',
    )

class SequentialTagger(object):
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.define_model()

    def define_model(self):
        def create_recurrent_layer():
            if self.config.cell_type == 'LSTM':
                return LSTM(self.config.cell_size, return_sequences=True)
            elif self.config.cell_type == 'GRU':
                return GRU(self.config.cell_size, return_sequences=True)
            raise ValueError("Unknown cell type: {}".format(self.config.cell_type))

        input_layer = Input(batch_shape=(None, self.dataset.maxlen), dtype='int8')
        emb = Embedding(len(self.dataset.vocab_x), self.config.embedding_size)(input_layer)
        layer = Masking(mask_value=0.)(emb)
        for _ in range(self.config.layers):
            if self.config.bidirectional:
                layer = Bidirectional(create_recurrent_layer())(layer)
        mlp = TimeDistributed(Dense(len(self.dataset.vocab_y), activation='softmax'))(layer)
        self.model = Model(inputs=input_layer, outputs=mlp)
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy')

    def run_train_test(self):
        ea = EarlyStopping(monitor='val_loss')
        tb = TensorBoard(log_dir='./logs')
        self.model.fit(
            self.dataset.x_train, self.dataset.y_train,
            epochs=10000,
            batch_size=self.config.batch_size,
            validation_split=0.2,
            callbacks=[ea, tb],
        )
        print(self.model.evaluate(self.dataset.x_test, self.dataset.y_test))


def main():
    args = parse_args()
    cfg = Config(args)
    dataset = DataSet(stdin)
    model = SequentialTagger(dataset, cfg)
    model.run_train_test()


if __name__ == '__main__':
    main()

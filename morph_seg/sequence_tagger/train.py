#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin
import os
from six.moves import cPickle
from datetime import datetime

import numpy as np
import pandas as pd

from keras.layers import Input, Dense, Embedding, Masking, \
        Bidirectional, Dropout, Conv1D
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, TensorBoard

from morph_seg.sequence_tagger.data import DataSet


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
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--log-results', action='store_true',
                   help="Write experiment statistics including"
                   "results to pandas dataframe")
    p.add_argument('--dataframe-path', type=str, default='results.tsv',
                   help="Path to results dataframe")
    p.add_argument('--save-model-dir', type=str, default=None,
                   help="Save trained model to HDF5 file")
    return p.parse_args()


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
        'log_dir': './logs',  # a subdir is created automatically
        'dataframe_path': 'results.tsv',
        'batch_size': 1024,
        'optimizer': 'Adam',
        'log_results': False,
        'log_tensorboard': False,
        'save_model_dir': None,
        'layers': 1,
        'bidirectional': True,
        'patience': 0,
    }
    __slots__ = tuple(defaults.keys()) + (
        'cell_type', 'cell_size', 'embedding_size',
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
        if self.log_tensorboard is True:
            self.generate_logdir()

    def generate_logdir(self):
        """Create and empty subdirectory for Tensorboard.
        The directory is created in the directory specified
        by log_dir. The first unused 4-digit number is used.
        """
        i = 0
        while os.path.exists(os.path.join(self.log_dir, "{0:04d}".format(i))):
            i += 1
        self.log_dir = os.path.join(self.log_dir, "{0:04d}".format(i))
        os.makedirs(self.log_dir)


class Result(DictConvertible):
    """POD class for experiment results."""

    __slots__ = (
        'train_acc', 'train_loss',
        'val_loss',
        'test_acc', 'test_loss',
        'running_time',
    )


class SequenceTagger(object):
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
            raise ValueError("Unknown cell type: {}".format(
                self.config.cell_type))

        input_layer = Input(batch_shape=(None, self.dataset.maxlen),
                            dtype='int8')
        emb = Embedding(len(self.dataset.x_vocab),
                        self.config.embedding_size)(input_layer)
        layer = Masking(mask_value=0.)(emb)
        for _ in range(self.config.layers):
            if self.config.bidirectional:
                layer = Bidirectional(create_recurrent_layer())(layer)
        mlp = TimeDistributed(Dense(len(self.dataset.y_vocab),
                                    activation='softmax'))(layer)
        self.model = Model(inputs=input_layer, outputs=mlp)
        self.model.compile(optimizer=self.config.optimizer,
                           loss='categorical_crossentropy')

    def run_train_test(self):
        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=self.config.patience)]
        if self.config.log_tensorboard:
            callbacks.append(TensorBoard(log_dir=self.config.log_dir,
                                         histogram_freq=1))
        start = datetime.now()
        history = self.model.fit(
            self.dataset.x_train, self.dataset.y_train,
            epochs=50,
            batch_size=self.config.batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1,
        )
        self.result.running_time = (
            datetime.now() - start
        ).total_seconds()
        self.result.val_loss = history.history['val_loss']
        self.result.train_loss = history.history['loss']
        self.evaluate()
        self.save_model()

    def evaluate(self):
        res = self.model.evaluate(self.dataset.x_test, self.dataset.y_test)
        self.result.test_loss = res
        y_proba = self.model.predict(self.dataset.x_test)
        y_predicted = np.argmax(y_proba, axis=-1)
        y_test = np.argmax(self.dataset.y_test, axis=-1)
        mask = (y_test != 0)  # padding mask
        real_char = np.sum(mask)  # exclude padding
        correct = np.sum((np.equal(y_predicted, y_test) & mask))
        self.result.test_acc = correct / float(real_char)
        if self.config.log_results is True:
            self.log()

    def save_model(self):
        if self.config.save_model_dir is not None:
            model_fn = os.path.join(self.config.save_model_dir, 'model.hdf5')
            self.model.save(model_fn)
            d = self.to_dict()
            params_fn = os.path.join(self.config.save_model_dir,
                                     'params.cpk')
            with open(params_fn, 'wb') as f:
                cPickle.dump(d, f)

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


class CNNTagger(SequenceTagger):
    def define_model(self):
        input_layer = Input(batch_shape=(None, self.dataset.maxlen),
                            dtype='int8')
        layer = Embedding(len(self.dataset.x_vocab),
                          self.config.embedding_size)(input_layer)
        layer = Dropout(0.8)(layer)
        layer = Conv1D(30, 5, padding='same', activation='relu',
                       strides=1)(layer)
        # layer = MaxPooling1D(pool_size=2)(layer)
        layer = LSTM(self.dataset.maxlen, return_sequences=True)(layer)
        layer = TimeDistributed(Dense(len(self.dataset.y_vocab),
                                      activation='softmax'))(layer)
        self.model = Model(inputs=input_layer, outputs=layer)
        self.model.compile(optimizer=self.config.optimizer,
                           loss='categorical_crossentropy')


def main():
    args = parse_args()
    cfg = Config(vars(args))
    dataset = DataSet(stdin)
    model = CNNTagger(dataset, cfg)
    model.run_train_test()


if __name__ == '__main__':
    main()

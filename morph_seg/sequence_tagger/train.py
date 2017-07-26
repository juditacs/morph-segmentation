#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin, stdout
import os
from six.moves import cPickle
from datetime import datetime
import copy
import yaml
import logging

import numpy as np
import pandas as pd

from keras.layers import Input, Dense, Embedding, Masking, \
        Bidirectional, Dropout, Conv1D
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, TensorBoard

from morph_seg.sequence_tagger.data import DataSet
from morph_seg.config import Config


def parse_args():
    p = ArgumentParser()
    p.add_argument('-t', '--train-file', default=stdin)
    p.add_argument('-c', '--config', type=str,
                   help="Location of YAML config")
    p.add_argument('-p', '--parameters', type=str,
                   help="Manually specify parameters."
                   "This option allows overriding parameters"
                   "from the config file."
                   "Format: param1=val1,param2=val2")
    p.add_argument('-a', '--architecture', choices=['RNN', 'CNN'],
                   default='RNN')
    return p.parse_args()


class DictConvertible(object):
    __slots__ = tuple()

    def to_dict(self):
        return {param: getattr(self, param, None)
                for param in self.__class__.__slots__}


class LSTMConfig(Config):

    defaults = copy.deepcopy(Config.defaults)

    defaults.update({
        'is_training': True,
        'log_dir': './logs',  # a subdir is created automatically
        'dataframe_path': 'results.tsv',
        'batch_size': 1024,
        'optimizer': 'Adam',
        'log_results': False,
        'log_tensorboard': False,
        'num_layers': 1,
        'bidirectional': True,
        'patience': 0,
        'activation': 'softmax',
        'cell_type': 'LSTM',
        'cell_size': 16,
        'embedding_size': 15,
    })
    __slots__ = tuple(defaults.keys())

    def set_derivable_params(self):
        super().set_derivable_params()
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

    def write_to_yaml(self, filename):
        d = self.to_dict()
        ser = {}
        for k, v in d.items():
            if isinstance(v, float):
                ser[k] = float(v)
            elif isinstance(v, np.float64):
                ser[k] = float(v)
            elif isinstance(v, list):
                ser[k] = [float(i) for i in v]
            else:
                ser[k] = v
        with open(filename, 'w') as f:
            yaml.dump(ser, f)


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
                            dtype='int32')
        emb = Embedding(len(self.dataset.x_vocab),
                        self.config.embedding_size)(input_layer)
        layer = Masking(mask_value=0.)(emb)
        for _ in range(self.config.num_layers):
            if self.config.bidirectional:
                layer = Bidirectional(create_recurrent_layer())(layer)
            else:
                layer = create_recurrent_layer()(layer)
        mlp = TimeDistributed(Dense(len(self.dataset.y_vocab),
                                    activation=self.config.activation))(layer)
        self.model = Model(inputs=input_layer, outputs=mlp)
        self.model.compile(optimizer=self.config.optimizer,
                           loss='categorical_crossentropy')

    def run_train(self):
        callbacks = [EarlyStopping(monitor='val_loss',
                                   patience=self.config.patience)]
        if self.config.log_tensorboard:
            callbacks.append(TensorBoard(log_dir=self.config.log_dir,
                                         histogram_freq=1))
        start = datetime.now()
        if hasattr(self.dataset, 'x_dev'):
            val_x = self.dataset.x_dev
            val_y = self.dataset.y_dev
            history = self.model.fit(
                self.dataset.x_train, self.dataset.y_train,
                epochs=self.config.max_epochs,
                batch_size=self.config.batch_size,
                validation_data=(val_x, val_y),
                callbacks=callbacks,
                verbose=1,
            )
        else:
            history = self.model.fit(
                self.dataset.x_train, self.dataset.y_train,
                epochs=self.config.max_epochs,
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

    def run_train_test(self):
        self.run_train()
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
        logging.info("Saving everthing to directory {}".format(
            self.config.model_dir))
        if self.config.model_dir is not None:
            model_fn = os.path.join(self.config.model_dir, 'model.hdf5')
            self.model.save(model_fn)
            d = self.to_dict()
            params_fn = os.path.join(self.config.model_dir,
                                     'params.cpk')
            with open(params_fn, 'wb') as f:
                cPickle.dump(d, f)
            result_fn = os.path.join(self.config.model_dir, 'result.yaml')
            self.result.write_to_yaml(result_fn)
            config_fn = os.path.join(self.config.model_dir, 'config.yaml')
            self.config.save_to_yaml(config_fn)

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
        for param, val in self.dataset.to_dict().items():
            d['data.{}'.format(param)] = val
        for param, val in self.result.to_dict().items():
            d['result.{}'.format(param)] = val
        d['model.json'] = self.model.to_json()
        return d


class CNNConfig(LSTMConfig):

    defaults = copy.deepcopy(LSTMConfig.defaults)

    defaults.update({
        'dropout': 0.2,
        'layers': [
            {'filters': 30, 'kernel_size': 5, 'strides': 1,
             'padding': 'same', 'activation': 'relu'},
        ]
    })
    __slots__ = tuple(defaults.keys())


class CNNTagger(SequenceTagger):

    def define_model(self):
        input_layer = Input(batch_shape=(None, self.dataset.maxlen),
                            dtype='int32')
        layer = Embedding(len(self.dataset.x_vocab),
                          self.config.embedding_size)(input_layer)

        layer = Dropout(self.config.dropout)(layer)

        for lparams in self.config.layers:
            layer = Conv1D(**lparams)(layer)

        layer = LSTM(self.dataset.maxlen, return_sequences=True)(layer)
        layer = TimeDistributed(Dense(len(self.dataset.y_vocab),
                                      activation='softmax'))(layer)
        self.model = Model(inputs=input_layer, outputs=layer)
        self.model.compile(optimizer=self.config.optimizer,
                           loss='categorical_crossentropy')


def main():
    args = parse_args()
    if args.architecture == 'CNN':
        cfg = CNNConfig.load_from_yaml(
            args.config, train_file=args.train_file,
            param_str=args.parameters
        )
        dataset = DataSet(cfg, args.train_file)
        model = CNNTagger(dataset, cfg)
    else:
        cfg = LSTMConfig.load_from_yaml(
            args.config, train_file=args.train_file,
            param_str=args.parameters
        )
        dataset = DataSet(cfg, args.train_file)
        model = SequenceTagger(dataset, cfg)
    cfg.save_to_yaml(stdout)
    model.run_train_test()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

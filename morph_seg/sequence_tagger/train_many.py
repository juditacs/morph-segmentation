#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin, stdout
import random
import logging

from morph_seg.sequence_tagger.data import DataSet
from morph_seg.sequence_tagger.train import LSTMConfig, SequenceTagger, \
    CNNConfig, CNNTagger


def parse_args():
    p = ArgumentParser()
    p.add_argument('-t', '--train-file', default=stdin)
    p.add_argument('--rnn-config', default=None, type=str,
                   help="Location of RNN YAML config")
    p.add_argument('--cnn-config', default=None, type=str,
                   help="Location of CNN YAML config")
    p.add_argument('-N', type=int, default=1)
    return p.parse_args()


rnn_ranges = {
    'cell_type': ['LSTM', 'GRU'],
    #'cell_size': [8, 16, 32, 64, 128, 256],
    'cell_size': [512, 1024, 2048],
    'batch_size': [32, 64],
    'num_layers': [1],
    #'num_layers': [1, 2, 3],
    'bidirectional': [True],
    'embedding_size': [5, 10, 20, 30, 40],
}

cnn_ranges = {
    'cell_type': ['LSTM', 'GRU'],
    #'cell_size': [8, 16, 32, 64, 128, 256],
    'cell_size': [512, 1024],
    'num_layers': [1, 2, 3, 4, 5],
    'lstm_layers': [1, 2],
    'embedding_size': [20, 30, 40],
}

cnn_layer_ranges = {
    'filters': [5, 10, 30, 50, 100],
    'kernel_size': [1, 2, 5, 10],
    'strides': [1], #, 2, 5],
    'activation': ['relu', 'sigmoid']
}

def update_cnn_config_with_random(cfg):
    layers = []
    for _ in range(random.choice(cnn_ranges['num_layers'])):
        l = {'padding': 'same'}
        for p, prange in cnn_layer_ranges.items():
            l[p] = random.choice(prange)
        layers.append(l)
    cfg.layers = layers
    for p, prange in cnn_ranges.items():
        setattr(cfg, p, random.choice(prange))


def main():
    args = parse_args()
    for n in range(args.N):
        if args.rnn_config is not None:
            logging.info("Running RNN experiment {}/{}".format(n+1, args.N))
            cfg = LSTMConfig.load_from_yaml(
                args.rnn_config, train_file=args.train_file,
            )
            for p, prange in rnn_ranges.items():
                val = random.choice(prange)
                setattr(cfg, p, val)
            cfg.save_to_yaml(stdout)
            dataset = DataSet(cfg, args.train_file)
            model = SequenceTagger(dataset, cfg)
            model.run_train_test()

        if args.cnn_config is not None:
            logging.info("Running CNN experiment {}/{}".format(n+1, args.N))
            cfg = CNNConfig.load_from_yaml(
                args.cnn_config, train_file=args.train_file,
            )
            update_cnn_config_with_random(cfg)
            cfg.save_to_yaml(stdout)
            dataset = DataSet(cfg, args.train_file)
            model = CNNTagger(dataset, cfg)
            model.run_train_test()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

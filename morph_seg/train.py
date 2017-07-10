#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from __future__ import unicode_literals

from argparse import ArgumentParser
from sys import stdin
import gzip

from experiment import Seq2seqExperiment
from data import DataSet


def parse_args():
    p = ArgumentParser(description='Run baseline seq2seq experiments.')
    p.add_argument('train_file', type=str, default=stdin,
                   help="Plain text of gzip file containing the training"
                   "data. If not specified, STDIN is used")
    p.add_argument('-r', '--result-file', type=str,
                   help='Path to result table')
    p.add_argument('--cell-type', choices=['LSTM', 'GRU'],
                   default='LSTM')
    p.add_argument('--cell-size', type=int, default=16)
    p.add_argument('--embedding-size', type=int, default=20)
    p.add_argument('--early-stopping-patience', type=int, default=10)
    p.add_argument('--early-stopping-threshold', type=float, default=1e-3)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--save-test-output', dest='test_output', type=str,
                   default=None)
    p.add_argument('--save-model', type=str, default=None,
                   help="Save model to directory")
    return p.parse_args()


def main():
    args = parse_args()
    data = DataSet()
    if args.train_file:
        if args.train_file.endswith('.gz'):
            with gzip.open(args.train_file) as infile:
                data.read_data_from_stream(infile)
        else:
            with open(args.train_file) as infile:
                data.read_data_from_stream(infile)
    else:
        data.read_data_from_stream(stdin)
    data.vectorize_samples()
    data.split_train_valid_test()
    logging.info("Train data shape: encoder - {}, decoder - {}".format(
        data.data_enc_train.shape,
        data.data_dec_train.shape,
    ))
    logging.info("Valid data shape: encoder - {}, decoder - {}".format(
        data.data_enc_valid.shape,
        data.data_dec_valid.shape,
    ))
    logging.info("Test data shape: encoder - {}, decoder - {}".format(
        data.data_enc_test.shape,
        data.data_dec_test.shape,
    ))
    conf = {
        'cell_type': args.cell_type,
        'cell_size': args.cell_size,
        'embedding_size': args.embedding_size,
        'patience': args.early_stopping_patience,
        'val_loss_th': args.early_stopping_threshold,
    }
    exp = Seq2seqExperiment(data, args.result_file, model_dir=args.save_model,
                            conf=conf)
    logging.info("Starting experiment")
    exp.run(save_stats=False)
    logging.info('Test loss: {}'.format(exp.model.result['test_loss'][-1]))
    if args.test_output is not None:
        with open(args.test_output, 'w') as f:
            exp.save_test_output(f, include_test_input=True)

if __name__ == '__main__':
    import logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

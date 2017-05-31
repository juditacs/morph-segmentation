#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin, stdout

from experiment import Seq2seqExperiment
from data import DataSet


def parse_args():
    p = ArgumentParser(description='Run baseline seq2seq experiments.')
    p.add_argument('-r', '--result-file', type=str,
                   help='Path to result table')
    p.add_argument('-l', '--length-limit', type=int, default=0,
                   help='Filter longer words than length_limit')
    p.add_argument('--cell-type', choices=['LSTM', 'GRU'],
                   default='LSTM')
    p.add_argument('--cell-size', type=int, default=16)
    p.add_argument('--embedding-size', type=int, default=20)
    p.add_argument('--early-stopping-patience', type=int, default=10)
    p.add_argument('--early-stopping-threshold', type=float, default=1e-3)
    p.add_argument('--save-test-output', dest='test_output', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    data = DataSet()
    data.read_data_from_stream(stdin, limit=50000,
                               length_limit=args.length_limit)
    data.vectorize_samples()
    data.split_train_valid_test()
    logging.info(str(data.data_dec_test.shape))
    conf = {
        'cell_type': args.cell_type,
        'cell_size': args.cell_size,
        'embedding_size': args.embedding_size,
        'patience': args.early_stopping_patience,
        'val_loss_th': args.early_stopping_threshold,
    }
    exp = Seq2seqExperiment(data, args.result_file, conf=conf)
    exp.run(save=False)
    if args.test_output is None:
        exp.save_test_output(stdout)
    else:
        with open(args.test_output, 'w') as f:
            exp.save_test_output(f)

if __name__ == '__main__':
    import logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

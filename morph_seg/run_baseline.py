#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin

from experiment import Seq2seqExperiment
from data import DataSet


def parse_args():
    p = ArgumentParser(description='Run baseline seq2seq experiments.')
    p.add_argument('-r', '--result-file', type=str,
                   help='Path to result table')
    p.add_argument('-n', type=int, default=1,
                   help='Run N number of experiments')
    p.add_argument('-l', '--length-limit', type=int, default=0,
                   help='Filter longer words than length_limit')
    return p.parse_args()


def main():
    args = parse_args()
    data = DataSet()
    data.read_data_from_stream(stdin, limit=50000,
                               length_limit=args.length_limit)
    data.vectorize_samples()
    data.split_train_valid_test()
    for n in range(args.n):
        print('EXPERIMENT {}'.format(n+1))
        exp = Seq2seqExperiment(data, args.result_file)
        print(exp.conf)
        exp.run()
        print(exp.model.result['test_loss'][-1])

if __name__ == '__main__':
    main()

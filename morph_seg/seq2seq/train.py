#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin

from morph_seg.seq2seq.config import Seq2seqConfig
from morph_seg.seq2seq.model import Seq2seqModel
from morph_seg.seq2seq.data import Seq2seqDataSet


def parse_args():
    """Set up and parse arguments"""
    p = ArgumentParser()
    p.add_argument('-t', '--train-file', default=stdin)
    p.add_argument('-c', '--config', type=str,
                  help="Location of YAML config")
    p.add_argument('-p', '--parameters', type=str,
                   help="Manually specify parameters."
                   "This option allows overriding parameters"
                   "from the config file."
                   "Format: param1=val1,param2=val2")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Seq2seqConfig.load_from_yaml(args.config, train_file=args.train_file,
                                       param_str=args.parameters)
    #print(cfg)
    dataset = Seq2seqDataSet(cfg, args.train_file)
    model = Seq2seqModel(cfg, dataset)
    model.run_train_test()

if __name__ == '__main__':
    import logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin, stdout
import logging

from morph_seg.sequence_tagger.train import CNNConfig, LSTMConfig, \
        CNNTagger, SequenceTagger
from morph_seg.sequence_tagger.data import TrainValidData


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
    p.add_argument('--prefix', type=str)
    return p.parse_args()


def main():
    args = parse_args()
    if args.architecture == 'CNN':
        cfg = CNNConfig.load_from_yaml(
            args.config, train_file=args.train_file,
            param_str=args.parameters
        )
        dataset = TrainValidData(cfg, args.prefix)
        model = CNNTagger(dataset, cfg)
    else:
        cfg = LSTMConfig.load_from_yaml(
            args.config, train_file=args.train_file,
            param_str=args.parameters
        )
        dataset = TrainValidData(cfg, args.prefix)
        model = SequenceTagger(dataset, cfg)
    cfg.save_to_yaml(stdout)
    model.run_train()
    model.save_model()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

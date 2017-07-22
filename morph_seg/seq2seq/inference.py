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

from morph_seg.seq2seq.config import Seq2seqInferenceConfig
from morph_seg.seq2seq.model import Seq2seqInferenceModel
from morph_seg.seq2seq.data import Seq2seqInferenceDataSet


def parse_args():
    """Set up and parse arguments"""
    p = ArgumentParser()
    p.add_argument('-t', '--test-file', default=stdin)
    p.add_argument('-m', '--model-dir', type=str,
                  help="Location of model directory")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_fn = os.path.join(args.model_dir, 'config.yaml')
    cfg = Seq2seqInferenceConfig.load_from_yaml(cfg_fn)
    #print(cfg)
    dataset = Seq2seqInferenceDataSet(cfg, args.test_file)
    model = Seq2seqInferenceModel(cfg, dataset)
    model.run_inference()

if __name__ == '__main__':
    import logging
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

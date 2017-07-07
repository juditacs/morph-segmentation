#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from __future__ import unicode_literals

from argparse import ArgumentParser
import os
from sys import stdin, stdout
import json

from data import EncoderInput
from model import SimpleSeq2seq


def parse_args():
    p = ArgumentParser()
    p.add_argument('-m', '--model-dir', type=str, required=True,
                   help="Location of model directory.")
    p.add_argument('--maxlen-enc', type=int, default=0,
                   help="Maximum input allowed")
    p.add_argument('--maxlen-dec', type=int, default=0,
                   help="Maximum output allowed")
    p.add_argument('--cell-type', choices=['LSTM', 'GRU'],
                   default='LSTM')
    p.add_argument('--cell-size', type=int, default=16)
    p.add_argument('--embedding-size', type=int, default=20)
    p.add_argument('--layers', type=int, default=1,
                   help="Number of LSTM/GRU layers")
    return p.parse_args()


def main():
    args = parse_args()
    data = EncoderInput(args.model_dir)
    data.read_data_from_stream(stdin)
    data.vectorize_samples(frozen=True)
    model_params = os.path.join(args.model_dir, 'model_params.json')
    with open(model_params) as f:
        conf = json.load(f)
    model = SimpleSeq2seq(**conf)
    model.create_model(data)
    model.run_inference(data, os.path.join(args.model_dir, 'model'))
    model.save_test_output(stdout, include_test_input=False)

if __name__ == '__main__':
    main()

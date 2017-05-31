#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import random
import pandas as pd
from os import path

from model import SimpleSeq2seq


class Seq2seqExperiment(object):
    default_ranges = {
        'cell_type': ('LSTM', 'GRU'),
        'cell_size': (16, 32, 64, 128, 256, 512),
        'embedding_size': tuple(range(1, 30)),
    }
    def __init__(self, dataset, result_fn, conf=None, custom_pranges=None):
        self.dataset = dataset
        self.result_fn = result_fn
        self.custom_pranges = custom_pranges
        self.result = {}
        self.conf = conf
        self.generate_random_config = conf is None
        self.create_model()

    def create_model(self):
        if self.generate_random_config is True:
            conf = Seq2seqExperiment.generate_config(self.custom_pranges)
        else:
            conf = self.conf
        self.model = SimpleSeq2seq(conf['cell_type'], conf['cell_size'],
                                   conf['embedding_size'])
        self.model.create_model(self.dataset)
        self.conf = conf
        
    def to_dict(self):
        d = {}
        for param, val in self.conf.items():
            d['conf.{}'.format(param)] = val
        for param, val in self.model.result.items():
            d['result.{}'.format(param)] = val
        for param, val in self.dataset.to_dict().items():
            d['data.{}'.format(param)] = val
        return d

    def run(self, save=True, save_output_fn=None):
        self.model.train_and_test(self.dataset, batch_size=1000)
        if save:
            self.save()
        if save_output_fn is not None:
            self.model.save_test_output(save_output_fn)

    def save(self):
        d = self.to_dict()
        if not path.exists(self.result_fn):
            df = pd.DataFrame(columns=d.keys())
        else:
            df = pd.read_table(self.result_fn)
        df = df.append(d, ignore_index=True)
        df.sort_index(axis=1).to_csv(self.result_fn, sep='\t', index=False)

    @staticmethod
    def generate_config(ranges=None):
        if ranges is not None:
            r = Seq2seqExperiment.default_ranges.copy()
            r.update(ranges)
        else:
            r = Seq2seqExperiment.default_ranges
        conf = {}
        for param, prange in r.items():
            conf[param] = random.choice(prange)
        return conf

    def save_train_output(self, stream):
        self.model.save_train_output(stream)

    def save_test_output(self, stream):
        self.model.save_test_output(stream)



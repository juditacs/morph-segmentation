#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from morph_seg.seq2seq.model import Seq2seqModel


class Seq2seqExperiment(object):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.model = Seq2seqModel(config, dataset)

    def run_train_test(self):
        pass

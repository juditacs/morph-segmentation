#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from morph_seg.result import Result


class Seq2seqResult(Result):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

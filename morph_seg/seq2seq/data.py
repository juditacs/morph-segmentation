#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from morph_seg.data import DataSet


class Seq2seqDataSet(DataSet):

    def pad_sample(self, enc, dec):
        enc = enc + ['PAD'] * (self.maxlen_enc-len(enc))
        dec = ['GO'] + dec + ['STOP'] + ['PAD'] * (
            self.maxlen_dec-len(dec)-2)
        return enc, dec
    
    def is_too_long(self, enc, dec):
        return self.config.derive_maxlen is False and \
           (len(enc) > self.config.maxlen_enc or
            len(dec) > self.config.maxlen_dec-2)  # GO and STOP symbols

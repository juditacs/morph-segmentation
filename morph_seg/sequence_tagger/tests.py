#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import unittest
import io

from morph_seg.sequence_tagger.data import DataSet

train_input = u"""alma	BEEE
autót	BEEEB"""

test_input = u"""
alma
autót"""

class DataSetTest(unittest.TestCase):
    def test_train_read_shapes(self):
        d = DataSet(io.StringIO(train_input))
        self.assertEqual(d.x.shape, (2, 5))
        self.assertEqual(d.y.shape, (2, 5, 3))

if __name__ == '__main__':
    unittest.main()

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import unittest
import io

from morph_seg.evaluation.boundary_detection import compute_stats
from morph_seg.evaluation.morpheme_detection import compute_morph_detection_stats

correct_input = u"""ab cd	ab cd
e fgh	e fgh"""

several = u"""ab cd	a b cd
e fgh	ef gh
e fgh	e fgh"""

short_almost_correct = u"""ab cd	ab c d"""

input_lendiff = u"""ab cd	aab cd
ab cd	aaa b cd"""

input_shorter_output = u"""ab cd	ad"""


class MorphDetectionTest(unittest.TestCase):

    def test_long(self):
        stats = compute_morph_detection_stats(io.StringIO(several))
        self.assertEqual(stats['fp'], 4)
        self.assertEqual(stats['fn'],  3)
        self.assertEqual(stats['tp'], 3)
    def test_correct(self):
        stats = compute_morph_detection_stats(io.StringIO(correct_input))
        self.assertEqual(stats['precision'], 1.0)
        self.assertEqual(stats['recall'], 1.0)
        self.assertEqual(stats['F-score'], 1.0)

    def test_short_almost_correct(self):
        stats = compute_morph_detection_stats(
            io.StringIO(short_almost_correct))
        self.assertEqual(stats['precision'], 1.0/3)
        self.assertEqual(stats['recall'], 0.5)

    def test_word_average(self):
        stats = compute_morph_detection_stats(
            io.StringIO(input_lendiff), word_average=True)
        self.assertEqual(stats['precision'], (0.5+1.0/3)/2)
        self.assertEqual(stats['recall'], (0.5+0.5)/2)

class BoundaryEvalTest(unittest.TestCase):
    def test_correct(self):
        stats = compute_stats(io.StringIO(correct_input))
        self.assertEqual(stats['precision'], 1.0)
        self.assertEqual(stats['recall'], 1.0)
        self.assertEqual(stats['accuracy'], 1.0)
        self.assertEqual(stats['F-score'], 1.0)

    def test_short_almost_correct(self):
        stats = compute_stats(io.StringIO(short_almost_correct))
        self.assertEqual(stats['precision'], 0.5)
        self.assertEqual(stats['recall'], 1.0)
        self.assertEqual(stats['accuracy'], 2.0/3)
        self.assertEqual(stats['F-score'], 2.0/3)

    def test_lendiff_match_from_end(self):
        stats = compute_stats(io.StringIO(input_lendiff), match_from_start=False)
        self.assertEqual(stats['tp'], 2)
        self.assertEqual(stats['tn'], 3)
        self.assertEqual(stats['fp'], 1)
        self.assertEqual(stats['fn'], 0)

    def test_lendiff_match_from_start(self):
        stats = compute_stats(io.StringIO(input_lendiff), match_from_start=True)
        self.assertEqual(stats['tp'], 0)
        self.assertEqual(stats['tn'], 2)
        self.assertEqual(stats['fp'], 2)
        self.assertEqual(stats['fn'], 2)

    def test_lendiff_shorter(self):
        stats = compute_stats(io.StringIO(input_shorter_output))
        self.assertEqual(stats['tp'], 0)
        self.assertEqual(stats['tn'], 2)
        self.assertEqual(stats['fp'], 0)
        self.assertEqual(stats['fn'], 1)

if __name__ == '__main__':
    unittest.main()

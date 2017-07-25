#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import json


class Token(object):
    def __init__(self, fields):
        if len(fields) < 4:
            raise ValueError("Not enough fields for token")
        self.word = fields[0].lower()
        self.lemma = fields[1].lower()
        self.analysis = fields[2]
        self.full_analysis = json.loads(fields[3])

    def __hash__(self):
        return hash(self.word)

    def __eq__(self, other):
        return self.word == other.word

    def __str__(self):
        return '{}\t{}\t{}\t{}'.format(self.word, self.lemma, self.analysis, self.full_analysis)

    @classmethod
    def from_line(cls, line):
        return Token(line.strip().split('\t'))

    def has_low_vowel_lengthening(self):
        return len(self.lemma) > 2 and \
                self.lemma not in self.word and \
                self.lemma[:-1] in self.word

    def is_instrumental(self):
        return '[Ins]' in self.analysis

    def lemma_change(self):
        return self.lemma not in self.word

    predicates = {
        'low_vowel_lengthening': has_low_vowel_lengthening,
        'instrumental': is_instrumental,
        'lemma_change': lemma_change,
    }

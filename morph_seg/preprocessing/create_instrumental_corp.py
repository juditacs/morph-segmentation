#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import json
import gzip
from collections import defaultdict
from sys import stderr


def parse_args():
    p = ArgumentParser()
    p.add_argument('input_file', nargs='+', type=str)
    return p.parse_args()


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
    'low_vowel_lengthening': Token.has_low_vowel_lengthening,
    'instrumental': Token.is_instrumental,
    'lemma_change': Token.lemma_change,
}


def collect_corpus_stats(input_files):
    corp = defaultdict(set)
    all_words = set()
    for i, infile in enumerate(input_files):
        stderr.write('{}/{} {}\n'.format(i+1, len(input_files), infile))
        with gzip.open(infile, 'rt') as f:
            for line in f:
                try:
                    token = Token.from_line(line)
                except ValueError:
                    continue
                for typ, pred in predicates.items():
                    if pred(token) is True:
                        corp[typ].add(token)
                all_words.add(token)
    return corp, all_words


def collect_corpus(input_files, keep_word):
    corp = set()
    for i, infile in enumerate(input_files):
        stderr.write('{}/{} {}\n'.format(i+1, len(input_files), infile))
        with gzip.open(infile, 'rt') as f:
            for line in f:
                try:
                    token = Token.from_line(line)
                except ValueError:
                    continue
                if keep_word(token):
                    corp.add(token)
    return corp


def instrumental(word):
    return word.analysis == '[/N][Ins]' and \
            len(word.word) != len(word.lemma)

def main():
    args = parse_args()
    corp = collect_corpus(args.input_file, instrumental)
    for word in corp:
        print("{}\t{}".format(word.lemma, word.word))
    #for k, v in corp.items():
        #print(k, len(v), float(len(v)) / len(all_words))
    #print(len(all_words))

    #for token in corp['lemma_change'] - corp['low_vowel_lengthening'] - corp['instrumental']:
        #print(token)

if __name__ == '__main__':
    main()

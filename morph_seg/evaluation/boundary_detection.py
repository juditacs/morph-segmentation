#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from sys import stdin
from collections import defaultdict
from argparse import ArgumentParser, RawDescriptionHelpFormatter


def parse_args():
    p = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description="""Compute segmentation statistics.

        Input data is read from STDIN. Expected format:
        <gold segmentation> TAB <segmentation to evaluate>
        """
    )
    p.add_argument('-v', '--verbose', action='store_true',
                   help="Output verbose statistics")
    p.add_argument('--markdown', action='store_true',
                   help='Print markdown table')
    p.add_argument('--match-from-start', action='store_true',
                   help="Match from word beginning if the gold"
                   "and the guess have different length."
                   "They are matched from the end otherwise.")
    return p.parse_args()


def compute_stats(stream, match_from_start=False):
    stats = defaultdict(int)
    for line in stream:
        try:
            gold, guess  = line.decode('utf8').rstrip('\n').split('\t')[:2]
        except AttributeError:
            gold, guess = line.rstrip('\n').split('\t')[:2]
        guess = match_words(gold, guess, match_from_start, stats)
        update_stats(gold, guess, stats)
    compute_summary(stats)
    return stats


def compute_summary(stats):
    n_samples = 0
    for key in ('tp', 'tn', 'fp', 'fn'):
        n_samples += stats[key]
    prec = expr_or_zero(lambda: float(stats['tp']) / (stats['tp']+stats['fp']))
    rec = expr_or_zero(lambda: float(stats['tp']) / (stats['tp']+stats['fn']))
    acc = expr_or_zero(lambda: float(stats['tp']+stats['tn']) / n_samples)
    fscore = expr_or_zero(lambda: 2 * prec * rec / (prec+rec))
    stats['precision'] = prec
    stats['recall'] = rec
    stats['accuracy'] = acc
    stats['F-score'] = fscore


def expr_or_zero(expr):
    try:
        return expr()
    except ZeroDivisionError:
        return 0


def update_stats(left, right, stats):
    lsplits = collect_splits(left)
    rsplits = collect_splits(right)
    llen = len(left.replace(' ', ''))
    for boundary in range(llen-1):
        if boundary in lsplits:
            if boundary in rsplits:
                stats['tp'] += 1
            else:
                stats['fn'] += 1
        else:
            if boundary in rsplits:
                stats['fp'] += 1
            else:
                stats['tn'] += 1
    return stats

def match_words(left, right, from_start, stats):
    lw = left.replace(' ', '')
    rw = right.replace(' ', '')
    if lw != rw:
        stats['word_changed'] += 1
    if len(lw) == len(rw):
        return right
    stats['len_diff'] += 1
    out = []
    if from_start is False:
        left = left[::-1]
        right = right[::-1]
    ri = 0
    for i in range(min((len(rw), len(lw)))):
        if right[ri] == ' ':
            out.append(' ')
            ri += 1
        out.append(right[ri])
        ri += 1
    right = ''.join(out)
    if from_start is False:
        left = left[::-1]
        right = right[::-1]
    return right


def collect_splits(word):
    splits = set()
    idx = 0
    for c in word:
        if c == ' ':
            splits.add(idx-1)
        else:
            idx += 1
    return splits


def print_table(stats, columns):
    print('\n'.join(
        '{}\t{}'.format(k, stats.get(k, 0)) for k in columns
    ))
    
def print_markdown_table(stats, columns):
    print('|  Metric/stat | Value |')
    print('| ----- | ----- |')
    print('\n'.join(
        '| {} | {} | '.format(k, stats.get(k, 0)) for k in columns
    ))

def list_columns(verbose):
    columns = []
    if verbose:
        columns.extend(['tp', 'tn', 'fp', 'fn', 'word_changed', 'len_diff'])
    columns.extend(['precision', 'recall', 'accuracy', 'F-score'])
    return columns


def main():
    args = parse_args()
    stats = compute_stats(stdin, match_from_start=args.match_from_start)
    columns = list_columns(args.verbose)
    if args.markdown:
        print_markdown_table(stats, columns)
    else:
        print_table(stats, columns)

if __name__ == '__main__':
    main()

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin
from collections import defaultdict


def parse_args():
    p = ArgumentParser()
    p.add_argument('--word-avg', action='store_true',
                   help="Compute word average instead of"
                   "individual morphemes")
    p.add_argument('--markdown', action='store_true',
                   help='Print markdown table')
    p.add_argument('-v', '--verbose', action='store_true',
                   help="Output verbose statistics")
    return p.parse_args()


def expr_or_zero(expr):
    try:
        return expr()
    except ZeroDivisionError:
        return 0


def compute_morph_detection_stats(stream, word_average=False):
    if word_average:
        return compute_word_average_stats(stream)
    return compute_global_stats(stream)


def compute_global_stats(stream):
    stats = defaultdict(int)
    for line in stream:
        try:
            gold, guess  = line.decode('utf8').rstrip('\n').split('\t')[:2]
        except AttributeError:
            gold, guess = line.rstrip('\n').split('\t')[:2]
        lmorphs = set(gold.split())
        rmorphs = set(guess.split())
        stats['tp'] += len(lmorphs & rmorphs)
        stats['fp'] = len(rmorphs - lmorphs)
        stats['fn'] = len(lmorphs - rmorphs)
    compute_summary(stats)
    return stats


def compute_summary(stats):
    prec = expr_or_zero(lambda: float(stats['tp']) /
                        (stats['tp']+stats['fp']))
    rec = expr_or_zero(lambda: float(stats['tp']) / (stats['tp']+stats['fn']))
    fscore = expr_or_zero(lambda: 2 * prec * rec / (prec+rec))
    stats['precision'] = prec
    stats['recall'] = rec
    stats['F-score'] = fscore


def compute_word_average_stats(stream):
    precision = []
    recall = []
    fscore = []
    for line in stream:
        try:
            gold, guess  = line.decode('utf8').rstrip('\n').split('\t')[:2]
        except AttributeError:
            gold, guess = line.rstrip('\n').split('\t')[:2]
        lmorphs = set(gold.split())
        rmorphs = set(guess.split())
        tp = len(lmorphs & rmorphs)
        fp = len(rmorphs - lmorphs)
        fn = len(lmorphs - rmorphs)
        prec = expr_or_zero(lambda: float(tp) / (tp + fp))
        rec = expr_or_zero(lambda: float(tp) / (tp + fn))
        fs = expr_or_zero(lambda: 2 * prec * rec / (prec + rec))
        precision.append(prec)
        recall.append(rec)
        fscore.append(fs)
    return {
        'precision': sum(precision) / len(precision),
        'recall': sum(recall) / len(recall),
        'F-score': sum(fscore) / len(fscore),
    }


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
        columns.extend(['tp', 'fp', 'fn'])
    columns.extend(['precision', 'recall', 'F-score'])
    return columns

def main():
    args = parse_args()
    stats = compute_morph_detection_stats(stdin, args.word_avg)
    columns = list_columns(args.verbose)
    if args.markdown:
        print_markdown_table(stats, columns)
    else:
        print_table(stats, columns)

if __name__ == '__main__':
    main()

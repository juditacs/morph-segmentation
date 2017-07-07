#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin


def parse_args():
    p = ArgumentParser(description="Filter words that appear in a list")
    p.add_argument('filter', type=str)
    p.add_argument('--keep', action='store_true',
                   help="Keep only those words that appear in the list")
    p.add_argument('--column', type=int, default=1,
                   help="Filter Nth column")
    return p.parse_args()


def main():
    args = parse_args()
    column = args.column - 1
    with open(args.filter) as f:
        filt = set(l.split('\t')[0].strip() for l in f)
    for line in stdin:
        word = line.rstrip('\n').split('\t')[column]
        if args.keep is True and word not in filt:
            continue
        if args.keep is False and word in filt:
            continue
        print(line.rstrip('\n'))

if __name__ == '__main__':
    main()

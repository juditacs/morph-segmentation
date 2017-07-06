#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin


def parse_args():
    p = ArgumentParser()
    p.add_argument('columns', nargs='*', type=int,
                   help="Specify which columns should be reversed."
                   "All columns are reversed if not specified")
    p.add_argument('-d', '--delimiter', type=str, default='\t')
    return p.parse_args()


def reverse_columns_stdin(columns, delimiter):
    for line in stdin:
        try:
            line = line.decode('utf8')
        except AttributeError:
            pass
        fd = line.rstrip('\n').split(delimiter)
        out = []
        for i, field in enumerate(fd):
            if not columns or i+1 in columns:
                out.append(field[::-1])
            else:
                out.append(field)
        try:
            print(delimiter.join(out).encode('utf8'))
        except AttributeError:
            print(delimiter.join(out))


def main():
    args = parse_args()
    reverse_columns_stdin(args.columns, args.delimiter)

if __name__ == '__main__':
    main()

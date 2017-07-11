#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin, stdout


def parse_args():
    p = ArgumentParser()
    p.add_argument('--tagging-type', choices=['BE', 'BEMS'], default='BE',
                   help="Use 2 or 4 tag tagset.")
    return p.parse_args()


def tag_stream(instream, outstream, tagging_type):
    for line in instream:
        try:
            line = line.decode('utf8')
        except AttributeError:
            pass
        fd = line.rstrip('\n').split('\t')
        segmented = fd[-1]
        segments = []
        for segment in segmented.split(' '):
            if tagging_type == 'BE':
                segments.append('B{}'.format('E' * (len(segment)-1)))
            elif tagging_type == 'BEMS':
                if len(segment) == 1:
                    segments.append('S')
                else:
                    segments.append('B{}E'.format('M' * (len(segment)-2)))
        try:
            outstream.write(u'{}\t{}\n'.format('\t'.join(fd[:-1]),
                                               ''.join(segments)).encode('utf8'))
        except TypeError:
            outstream.write('{}\t{}\n'.format('\t'.join(fd[:-1]),
                                              ''.join(segments)))


def main():
    args = parse_args()
    tag_stream(stdin, stdout, tagging_type=args.tagging_type)

if __name__ == '__main__':
    main()

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
    p.add_argument('--columns', type=str, default="-1",
                   help="Column to convert to BE tagging."
                   "More columns may be specified separated by comma."
                   "By default only the last column is converted")
    p.add_argument('--conll', action='store_true',
                   help="Convert to one sample-per-line (CoNLL-like) format."
                   "Words are delimited by empty lines."
                   "If true, only the first two columns are kept and"
                   "the rest are discarded.")
    return p.parse_args()


def tag_stream(instream, outstream, tagging_type, columns, conll=False):
    columns = set(int(c) for c in columns.split(','))
    for line in instream:
        try:
            line = line.decode('utf8')
        except AttributeError:
            pass
        fd = line.rstrip('\n').split('\t')
        output_fields = []
        for i, field in enumerate(fd):
            if i+1 in columns or (-1 in columns and i == len(fd) - 1):
                segments = []
                for segment in field.split(' '):
                    if tagging_type == 'BE':
                        segments.append('B{}'.format('E' * (len(segment)-1)))
                    elif tagging_type == 'BEMS':
                        if len(segment) == 1:
                            segments.append('S')
                        else:
                            segments.append('B{}E'.format('M' * (len(segment)-2)))
                output_fields.append(''.join(segments))
            else:
                output_fields.append(field)
        if conll:
            write_sample_line_by_line(output_fields, outstream)
        else:
            write_sample_one_line(output_fields, outstream)
        outstream.write('\n')


def write_sample_line_by_line(output_fields, outstream):
    sample, tagging = output_fields[:2]
    try:
        outstream.write('\n'.join(
            u'{}\t{}'.format(sample[i], tagging[i]) for i in range(len(sample))
        ).encode('utf8') + '\n')
    except TypeError:
        outstream.write('\n'.join(
            '{}\t{}'.format(sample[i], tagging[i]) for i in range(len(sample))
        ) + '\n')


def write_sample_one_line(output_fields, outstream):
    try:
        outstream.write(u'{}\n'.format(
            '\t'.join(output_fields)).encode('utf8'))
    except TypeError:
        outstream.write(u'{}\n'.format('\t'.join(output_fields)))


def main():
    args = parse_args()
    tag_stream(stdin, stdout, tagging_type=args.tagging_type, columns=args.columns,
               conll=args.conll)

if __name__ == '__main__':
    main()

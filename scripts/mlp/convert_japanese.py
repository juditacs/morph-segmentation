#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser


def parse_args():
    p = ArgumentParser()
    p.add_argument('orig', type=str, help="Path to orig file")
    p.add_argument('segd', type=str, help="Path to segd file")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.orig) as orig_f, open(args.segd) as segd_f:
        for orig_line in orig_f:
            orig_line = orig_line.rstrip('\n')
            si = 0
            seg_line = next(segd_f)
            output = []
            for i, c in enumerate(orig_line):
                if seg_line[si] == ' ':
                    output.append('B')
                    si += 1
                else:
                    output.append('E')
                si += 1
            print(''.join(output))
            if len(orig_line) != len(output):
                raise ValueError("Different line length")


if __name__ == '__main__':
    main()

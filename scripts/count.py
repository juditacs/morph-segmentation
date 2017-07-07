#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from sys import stdin
from collections import defaultdict


def main():
    freqs = defaultdict(int)
    for line in stdin:
        freqs[line.rstrip('\n')] += 1
    for k, v in sorted(freqs.items(), key=lambda x: -x[1]):
        print('{}\t{}'.format(k, v))

if __name__ == '__main__':
    main()

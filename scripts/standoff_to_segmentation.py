#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from sys import stdin


def main():
    for line in stdin:
        word, BE = line.rstrip("\n").split("\t")[:2]
        segments = []
        for i, c in enumerate(word):
            if BE[i] == "B" and i > 0:
                segments.append(" ")
            segments.append(c)
        print("{}\t{}".format(word, "".join(segments)))

if __name__ == '__main__':
    main()

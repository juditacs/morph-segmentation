#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from sys import stdin


def convert():
    for line in stdin:
        if not line.strip():
            continue
        words = line.rstrip('\n').split(' ')
        out = []
        for word in words:
            morphs = word.split("\\\\")
            tags = ['B{}'.format('E'*(len(morphs[0])-1))]
            tags.extend(
                ['M{}'.format('E'*(len(m)-1)) for m in morphs[1:]]
            )
            out.append(''.join(tags))
        print('_'.join(out))


if __name__ == '__main__':
    convert()

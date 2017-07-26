#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin

from morph_seg.sequence_tagger.inference import Inference


class MLPInference(Inference):
    def print_segmentation(self):
        for i, sample in enumerate(self.dataset.samples):
            out = []
            offset = len(self.decoded[i]) - len(sample)
            for j, s in enumerate(sample):
                if self.decoded[i][j+offset] == '_':
                    out.append(' ')
                elif self.decoded[i][j+offset] == 'M':
                    out.append("\\\\")
                out.append(s)
            print(''.join(out).strip())


def parse_args():
    p = ArgumentParser()
    p.add_argument('--model-dir', type=str, required=True,
                   help="Location of model and parameter files")
    return p.parse_args()


def main():
    args = parse_args()
    inf = MLPInference(args.model_dir, stdin)
    inf.run_inference()
    inf.print_segmentation()


if __name__ == '__main__':
    main()

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
from morph_seg.sequence_tagger.data import WindowInferenceData


class MLPInference(Inference):
    def __init__(self, model_dir, stream_or_file, window, step):
        self.dataset = WindowInferenceData(model_dir, stream_or_file, window, step)
        self.model_dir = model_dir
        self.load_model()

    def print_segmentation(self):
        outputs = []
        for i, sample in enumerate(self.dataset.samples):
            out = []
            offset = len(self.decoded[i]) - len(sample)
            for j, s in enumerate(sample):
                if self.decoded[i][j+offset] == 'M':
                    out.append("\\\\")
                out.append(s)
            # print(''.join(out).strip())
            outputs.append(''.join(out).strip())
        self.print_lookup_mapping(outputs)

    def print_lookup_mapping(self, outputs):
        if hasattr(self.dataset, 'mapping'):
            for i in range(0, len(self.dataset.mapping)-1):
                merged = ''.join(outputs[self.dataset.mapping[i]:self.dataset.mapping[i+1]])
                print(merged)
            merged = ''.join(outputs[self.dataset.mapping[-1]:])
            print(merged)


class WordSegmenterInference(MLPInference):
    def print_segmentation(self):
        outputs = []
        for i, sample in enumerate(self.dataset.samples):
            out = []
            offset = len(self.decoded[i]) - len(sample)
            for j, s in enumerate(sample):
                if self.decoded[i][j+offset] == 'B':
                    out.append(" ")
                out.append(s)
            #print(''.join(out).strip())
            outputs.append(''.join(out).strip())
        self.print_lookup_mapping(outputs)


class VietnameseInference(MLPInference):
    def print_segmentation(self):
        outputs = []
        for i, sample in enumerate(self.dataset.samples):
            out = []
            offset = len(self.decoded[i]) - len(sample)
            for j, s in enumerate(sample):
                if self.decoded[i][j+offset] == '1':
                    out.append('_')
                elif self.decoded[i][j+offset] == '2':
                    out.append('__')
                else:
                    out.append(s)
            #print(''.join(out).strip())
            outputs.append(''.join(out).strip())
        self.print_lookup_mapping(outputs)


def parse_args():
    p = ArgumentParser()
    p.add_argument('--model-dir', type=str, required=True,
                   help="Location of model and parameter files")
    p.add_argument('--type', choices=['morph', 'word', 'vietnamese'],
                   default='morph')
    p.add_argument('--window', type=int, default=100)
    p.add_argument('--step', type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    if args.type == 'morph':
        inf = MLPInference(args.model_dir, stdin, args.window, args.step)
    elif args.type == 'vietnamese':
        inf = VietnameseInference(args.model_dir, stdin, args.window, args.step)
    else:
        inf = WordSegmenterInference(args.model_dir, stdin, args.window, args.step)
    inf.run_inference()
    inf.print_segmentation()


if __name__ == '__main__':
    main()

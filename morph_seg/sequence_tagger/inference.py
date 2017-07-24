#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin
import os

import numpy as np
import keras

from morph_seg.sequence_tagger.data import InferenceData


def parse_args():
    p = ArgumentParser()
    p.add_argument('--model-dir', type=str, required=True,
                   help="Location of model and parameter files")
    return p.parse_args()


class Inference(object):
    def __init__(self, model_dir, stream_or_file):
        self.dataset = InferenceData(model_dir, stream_or_file)
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        model_fn = os.path.join(self.model_dir, 'model.hdf5')
        self.model = keras.models.load_model(model_fn)

    def run_inference(self):
        res = self.model.predict(self.dataset.x)
        labels = np.argmax(res, axis=-1)
        self.decoded = self.dataset.decode(labels)

    def print_segmentation(self):
        """Print segmentation output. """
        for i, sample in enumerate(self.dataset.samples):
            out = []
            offset = len(self.decoded[i]) - len(sample)
            for j, s in enumerate(sample):
                if self.decoded[i][j+offset] == 'B':
                    out.append(' ')
                out.append(s)
                print(''.join(out).strip())


def main():
    args = parse_args()
    inf = Inference(args.model_dir, stdin)
    inf.run_inference()
    inf.print_segmentation()


if __name__ == '__main__':
    main()

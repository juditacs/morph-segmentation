#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import yaml
from datetime import datetime

from morph_seg.utils import DictConvertible


class Result(DictConvertible):
    __slots__ = (
        'train_acc', 'train_loss',
        'val_acc', 'val_loss',
        'test_acc', 'test_loss',
        'start', 'end', 'running_time',
        'epochs_run', 'early_stopped',
    )
    def set_start(self):
        self.start = datetime.now()

    def set_end(self):
        self.end = datetime.now()
        self.running_time = (self.end-self.start).total_seconds()

    def save_to_yaml(self, filename):
        cfg = {}
        for key in self.__slots__:
            cfg[key] = getattr(self, key, None)
        with open(filename, 'w') as f:
            yaml.dump(cfg, f)

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import yaml
import os


class ConfigError(Exception): pass


class Config(object):
    defaults = {
        'delimiter': None,
        'vocab_path': None,
        'share_vocab': False,
        'frozen_vocab': False,
        'batch_size': 1024,
        'max_epochs': 10000,
        'early_stopping_threshold': 0.001,
        'patience': 2,
        'save_model': True,
        'model_dir': None,
    }

    __slots__ = tuple(defaults) + (
    )

    def __init__(self, cfg_dict=None, param_str=None, **kwargs):
        # load defaults first
        for param, value in Config.defaults.items():
            setattr(self, param, value)

        # override defaults with config dictionary
        for param, value in cfg_dict.items():
            setattr(self, param, value)

        # parse param_str and set values
        self.parse_and_set_param_str(param_str)

        # finally override params if specified in kwargs
        for param, value in kwargs.items():
            setattr(self, param, value)

        self.set_derivable_params()
        self.check_params()
        self.create_model_dir()

    def set_derivable_params(self):
        pass

    def parse_and_set_param_str(self, param_str):
        if param_str is None:
            return
        for param_val in param_str.split(','):
            param, value = param_val.split('=')
            try:
                value = float(value)
            except ValueError:
                pass
            setattr(self, param, value)

    def create_model_dir(self):
        if self.save_model is False:
            return
        i = 0
        path_fmt = '{0:04d}'
        while os.path.exists(os.path.join(self.model_dir, path_fmt.format(i))):
            i += 1
        self.model_dir = os.path.join(self.model_dir, path_fmt.format(i))
        os.makedirs(self.model_dir)

    def check_params(self):
        if self.save_model is True and self.model_dir is None:
            raise ConfigError("A model_dir must be specified if "
                              "save_model is True")

    def __repr__(self):
        return '{}'.format({k: getattr(self, k, None)
                for k in self.__class__.__slots__})

    @property
    def vocab_enc_path(self):
        if self.share_vocab:
            return self.vocab_path
        return '{}.enc'.format(self.vocab_path)

    @property
    def vocab_dec_path(self):
        if self.share_vocab:
            return self.vocab_path
        return '{}.dec'.format(self.vocab_path)

    @classmethod
    def load_from_yaml(cls, filename, param_str=None, **override_params):
        with open(filename) as f:
            cfg = yaml.load(f)
        return cls(cfg, param_str, **override_params)

    def save_to_yaml(self, filename):
        cfg = {}
        for key in self.__slots__:
            cfg[key] = getattr(self, key, None)
        with open(filename, 'w') as f:
            yaml.dump(cfg, f)

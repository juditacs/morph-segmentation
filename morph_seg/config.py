#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import yaml
import os


class ConfigError(Exception):
    pass


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
        'reverse_input': False,
        'reverse_output': False,
        'test_size': 10000,
        'train_file': None,
    }

    __slots__ = tuple(defaults) + (
        'train_file',
    )

    int_values = (
        'batch_size', 'max_epochs', 'patience', 'test_size',
    )

    def __init__(self, cfg_dict=None, param_str=None, **kwargs):
        # load defaults first
        for param, value in self.defaults.items():
            setattr(self, param, value)

        # override defaults with config dictionary
        if cfg_dict is not None:
            for param, value in cfg_dict.items():
                setattr(self, param, value)

        # parse param_str and set values
        self.parse_and_set_param_str(param_str)

        # finally override params if specified in kwargs
        for param, value in kwargs.items():
            setattr(self, param, value)

        self.set_derivable_params()
        self.check_params()

    def derive_and_create_model_dir(self):
        # model_dir needs to be fully defined
        if self.is_training is False:
            return
        i = 0
        path_fmt = '{0:04d}'
        while os.path.exists(os.path.join(self.model_dir, path_fmt.format(i))):
            i += 1
        self.model_dir = os.path.join(self.model_dir, path_fmt.format(i))
        os.makedirs(self.model_dir)

    def set_derivable_params(self):
        self.derive_and_create_model_dir()
        if isinstance(self.train_file, str):
            self.train_file = os.path.abspath(self.train_file)
        if self.vocab_path is None:
            self.vocab_path = os.path.join(self.model_dir, 'vocab')

    def parse_and_set_param_str(self, param_str):
        if param_str is None:
            return
        for param_val in param_str.split(','):
            param, value = param_val.split('=')
            if param in self.int_values:
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            setattr(self, param, value)

    def check_params(self):
        if self.is_training is False:
            model_path = os.path.join(self.model_dir, 'params.yaml')
            if not os.path.exists(model_path):
                raise ConfigError(
                    "Model directory must contain a valid saved model. "
                    "Directory: {}".format(self.model_dir)
                )

    def __repr__(self):
        return '{}'.format({k: getattr(self, k, None)
                            for k in self.__class__.__slots__})

    @property
    def vocab_enc_path(self):
        return '{}.enc'.format(self.vocab_path)

    @property
    def vocab_dec_path(self):
        return '{}.dec'.format(self.vocab_path)

    @classmethod
    def load_from_yaml(cls, filename, param_str=None, **override_params):
        with open(filename) as f:
            cfg = yaml.load(f)
        return cls(cfg, param_str, **override_params)

    def save_to_yaml(self, filename):
        cfg = {}
        for key in self.__slots__:
            if key == 'train_file':
                if isinstance(self.train_file, str):
                    cfg['train_file'] = self.train_file
                else:
                    cfg['train_file'] = 'stdin'
            else:
                cfg[key] = getattr(self, key, None)
        with open(filename, 'w') as f:
            yaml.dump(cfg, f)

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import copy

from morph_seg.config import Config, ConfigError


class Seq2seqConfig(Config):
    defaults = copy.deepcopy(Config.defaults)
    defaults.update({
        'derive_maxlen': True,
        'share_vocab': True,
        'bidirectional': False,
        'pad_left': False,
        'cell_type': 'LSTM',
        'cell_size': 63,
        'num_residual': 0,
        'num_layers': 1,
        'attention_type': None,
        'attention_layer_size': 16,
        'is_training': True,
        'batch_size': 16,
        'optimizer': 'AdamOptimizer',
        'optimizer_kwargs': {},
        'embedding_dim_enc': 0,
        'embedding_dim_dec': 0,
    })
    int_values = Config.int_values + (
        'maxlen_enc', 'maxlen_dec',
        'embedding_dim', 'embedding_dim_enc', 'embedding_dim_dec',
    )
    __slots__ = tuple(defaults) + int_values + Config.__slots__

    def set_derivable_params(self):
        super().set_derivable_params()
        if self.embedding_dim_enc == 0:
            self.embedding_dim_enc = self.embedding_dim
        if self.embedding_dim_dec == 0:
            self.embedding_dim_dec = self.embedding_dim
        if self.share_vocab:
            self.embedding_dim_enc = self.embedding_dim
            self.embedding_dim_dec = self.embedding_dim

    def check_params(self):
        if self.derive_maxlen is False:
            if self.maxlen_enc == 0 or self.maxlen_dec == 0:
                raise ConfigError(
                    "maxlen_enc and maxlen_dec should be "
                    "defined if derive_maxlen is False"
                )
        if self.num_residual > self.num_layers / 2:
            raise ConfigError(
                "The number of residual connection cannot be "
                "greater than layers / 2"
            )
        if self.attention_type not in ('luong', 'bahdanau'):
            raise ConfigError(
                "Attention type most be luong or bahdanau"
            )


class Seq2seqInferenceConfig(Seq2seqConfig):
    defaults = copy.deepcopy(Seq2seqConfig.defaults)
    defaults.update({
        'is_training': False,
    })

    def __init__(self, cfg_dict=None, param_str=None, **kwargs):
        self.__class__.defaults = Seq2seqInferenceConfig.defaults
        cfg_dict['is_training'] = False
        cfg_dict['save_model'] = False
        super().__init__(cfg_dict, param_str, **kwargs)

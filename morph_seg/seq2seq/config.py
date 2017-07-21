#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from morph_seg.config import Config, ConfigError


class Seq2seqConfig(Config):
    defaults = Config.defaults
    defaults.update({
        'derive_maxlen': True,
        'share_vocab': True,
        'bidirectional': False,
        'cell_type': 'LSTM',
        'cell_size': 64,
        'num_residual': 0,
        'num_layers': 1,
        'attention_type': None,
        'attention_layer_size': 16,
        'is_training': True,
        'batch_size': 16,
        'optimizer': 'AdamOptimizer',
        'optimizer_kwargs': {},
    })
    __slots__ = tuple(defaults) + Config.__slots__ + (
        'maxlen_enc', 'maxlen_dec',
        'embedding_dim', 'embedding_dim_enc', 'embedding_dim_dec',
    )

    def set_derivable_params(self):
        super().set_derivable_params()
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

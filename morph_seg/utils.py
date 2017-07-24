#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

class DictConvertible(object):
    __slots__ = tuple()

    def to_dict(self):
        return {param: getattr(self, param, None) for param in self.__slots__}

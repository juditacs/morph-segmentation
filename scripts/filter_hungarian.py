#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin
import re


hu_alphabet = 'abcdefghijklmnopqrstuvwxyzáéíóöőúüű'
punct_whitelist = '.-'

def parse_args():
    p = ArgumentParser()
    p.add_argument('--alphabet', type=str, default=hu_alphabet,
                   help='Filter lines if they contain anything outside'
                   'this alphabet')
    p.add_argument('--case-insensitive', action='store_true',
                   help='Turn off case sensitivity')
    p.add_argument('--punct-whitelist', type=str, default=punct_whitelist,
                   help='Allowed punctuation')
    p.add_argument('--allow-digits', action='store_true',
                   help='Allow digits in the input')
    p.add_argument('--contains-letter', action='store_true',
                   help='Line has to contain at least one letter from'
                   'the alphabet.')
    return p.parse_args()


def compile_filter_regex(alphabet, allow_digits, contains_letter, punct_whitelist, case_insensitive):
    digit_s = '0-9' if allow_digits else ''
    if case_insensitive:
        alphabet = '{0}{1}'.format(alphabet.lower(), alphabet.upper())
    if contains_letter:
        s = r'^[{0}{1}{2}\s]*[{0}][{0}{1}{2}\s]*$'.format(
            re.escape(alphabet),
            re.escape(punct_whitelist), digit_s)
    else:
        s = r'^[{0}{1}{2}\s]*$'.format(
            re.escape(alphabet),
            re.escape(punct_whitelist), digit_s)
    return re.compile(s, re.UNICODE)


def filter_stdin(filter_re):
    for line in stdin:
        if filter_re.match(line):
            print(line.rstrip('\n'))
    

def main():
    args = parse_args()
    filter_re = compile_filter_regex(
        args.alphabet,
        allow_digits=args.allow_digits,
        contains_letter=args.contains_letter,
        punct_whitelist=args.punct_whitelist,
        case_insensitive=args.case_insensitive
    )
    filter_stdin(filter_re)

if __name__ == '__main__':
    main()

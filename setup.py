#! /usr/bin/env python

from setuptools import find_packages, setup


setup(
    name='morph_seg',
    version='0.1.0',
    description="Morphological segmentation experiments",
    author='Judit Acs',
    author_email='judit@sch.bme.hu',
    packages=find_packages(),
    package_dir={'': '.'},
    provides=['morph_seg'],
)

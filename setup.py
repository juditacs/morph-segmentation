#! /usr/bin/env python


from distutils.core import setup


setup(
    name='morph_seg',
    version='0.1.0',
    description="Morphological segmentation experiments",
    author='Judit Acs',
    author_email='judit@sch.bme.hu',
    packages=['morph_seg'],
    package_dir={'': '.'},
    provides=['morph_seg'],
)

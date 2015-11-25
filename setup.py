#!/usr/bin/env python

from setuptools import setup

version = '0.6.0.dev'
setup(
    name='thunder-python',
    version=version,
    description='large-scale image and time series analysis',
    author='freeman-lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/thunder-project/thunder',
    packages=['thunder',
              'thunder.clustering',
              'thunder.factorization',
              'thunder.data',
              'thunder.data.fileio',
              'thunder.data.blocks',
              'thunder.data.series',
              'thunder.data.images',
              'thunder.regression',
              'thunder.regression.linear',
              'thunder.regression.nonlinear',
              'thunder.utils'
              ],
    package_data={'thunder.lib': ['thunder_python-' + version + '-py2.7.egg']},
    install_requires=open('requirements.txt').read().split(),
    long_description='See https://github.com/thunder-project/thunder'
)

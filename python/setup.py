#!/usr/bin/env python

from setuptools import setup

setup(name='thunder',
    version='0.1.0',
    description='Large-scale neural data analysis in Spark',
    author='The Freeman Lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/freeman-lab/thunder',
    packages=['thunder','thunder.regression','thunder.factorization','thunder.classification','thunder.clustering','thunder.timeseries','thunder.io','thunder.util','thunder.viz'],
    )
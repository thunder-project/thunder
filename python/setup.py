#!/usr/bin/env python

from setuptools import setup
import thunder

setup(name='Thunder',
    version=str(thunder.__version__),
    description='Large-scale neural data analysis in Spark',
    author='The Freeman Lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/freeman-lab/thunder',
    packages=['thunder','thunder.regression','thunder.factorization','thunder.classification','thunder.clustering','thunder.timeseries','thunder.io','thunder.util','thunder.viz'],
    long_description=open('../README.rst').read(),
    license=open('../LICENSE.txt'),
    install_requires=[open('requirements.txt').read()]
    )
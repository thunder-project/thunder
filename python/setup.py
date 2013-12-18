#!/usr/bin/env python

from setuptools import setup

setup(name='Thunder',
      version='1.0',
      description='Neural data analysis in Spark',
      author='The Freeman Lab',
      author_email='the.freeman.lab@gmail.com',
      url='https://github.com/freeman-lab/thunder',
      packages=['thunder','thunder.regression','thunder.factorization','thunder.summary','thunder.util'],
     )
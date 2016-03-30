#!/usr/bin/env python

from setuptools import setup

version = '1.0.0'

extra = {'all': ['thunder-regression', 'thunder-registration']}

setup(
    name='thunder-python',
    version=version,
    description='large-scale image and time series analysis',
    author='freeman-lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/thunder-project/thunder',
    packages=[
        'thunder',
        'thunder.blocks',
        'thunder.series',
        'thunder.images'
    ],
    install_requires=open('requirements.txt').read().split('\n'),
    extras_require=extra,
    long_description='See https://github.com/thunder-project/thunder'
)

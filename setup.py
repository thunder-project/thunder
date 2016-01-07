#!/usr/bin/env python

from setuptools import setup

version = '0.6.0.dev'

required = open('requirements.txt').read().split('\n')
extra = {'all': ['mist', 'rime']}

setup(
    name='thunder-python',
    version=version,
    description='large-scale image and time series analysis',
    author='freeman-lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/thunder-project/thunder',
    packages=[
        'thunder',
        'thunder',
        'thunder.blocks',
        'thunder.series',
        'thunder.images'
    ],
    package_data={'thunder.lib': ['thunder_python-' + version + '-py2.7.egg']},
    install_requires=required,
    extra_requires=extra,
    long_description='See https://github.com/thunder-project/thunder'
)

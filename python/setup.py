#!/usr/bin/env python

from setuptools import setup
import thunder

setup(
    name='thunder-python',
    version=str(thunder.__version__),
    description='Large-scale neural data analysis in Spark',
    author='The Freeman Lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/freeman-lab/thunder',
    packages=['thunder',
              'thunder.clustering',
              'thunder.decoding',
              'thunder.factorization',
              'thunder.imgprocessing',
              'thunder.imgprocessing.regmethods',
              'thunder.lib',
              'thunder.rdds',
              'thunder.rdds.fileio',
              'thunder.rdds.imgblocks',
              'thunder.regression',
              'thunder.standalone',
              'thunder.utils',
              'thunder.viz'],
    scripts = ['bin/thunder', 'bin/thunder-submit', 'bin/thunder-ec2'],
    package_data = {'thunder.utils': ['data/fish/bin/conf.json', 'data/fish/bin/*.bin', 'data/fish/tif-stack/*.tif', 'data/iris/conf.json', 'data/iris/iris.bin', 'data/iris/iris.mat', 'data/iris/iris.npy', 'data/iris/iris.txt'], 'thunder.lib': ['thunder_2.10-' + str(thunder.__version__) + '.jar']},
    long_description=open('README.rst').read(),
    install_requires=open('requirements.txt').read().split()
)

"""
Utilities for generating test datasets
"""

import os
from string import replace
from numpy import array, random, arange, ceil, floor, dot, shape, sqrt, arange, meshgrid, ones
from thunder.io import load


def appendkeys(data):
    data = array(data)
    n = shape(data)[0]
    x = (random.rand(n) * n).astype(int)
    y = (random.rand(n) * n).astype(int)
    z = (random.rand(n) * n).astype(int)
    data_zipped = zip(x, y, z, data)
    return map(lambda (k1, k2, k3, v): ((k1, k2, k3), v), data_zipped)


class DataSets(object):

    def __init__(self, sc):
        self.sc = sc
        self.path = os.path.dirname(os.path.realpath(__file__)).replace('python/thunder/io', 'data')

    @staticmethod
    def create(sc, name, **opts):
        return DATASET_MAKERS[name](sc).make(**opts)

    @staticmethod
    def load(sc, name, **opts):
        return DATASET_LOADERS[name](sc).fromfile()


class KMeansData(DataSets):

    def make(self, k, npartitions=10, ndims=5, nrecords=100, noise=0.1):
        random.seed(42)
        centers = random.randn(k, ndims)
        gen_func = lambda i: centers[int(floor(random.rand(1, 1) * k))] + noise*random.rand(ndims)
        data_local = map(gen_func, range(0, nrecords))
        self.centers = centers
        self.rdd = self.sc.parallelize(appendkeys(data_local), npartitions)
        return self


class PCAData(DataSets):

    def make(self, k, npartitions=10, nrows=100, ncols=10):
        random.seed(42)
        u = random.randn(nrows, k)
        v = random.randn(k, ncols)
        a = dot(u, v)
        a += random.randn(shape(a)[0], shape(a)[1])
        self.u = u
        self.v = v
        self.rdd = self.sc.parallelize(appendkeys(a), npartitions)
        return self


class FishData(DataSets):

    def fromfile(self):
        return load(self.sc, os.path.join(self.path, 'fish.txt'))


class IrisData(DataSets):

    def fromfile(self):
        return load(self.sc, os.path.join(self.path, 'iris.txt'))

DATASET_MAKERS = {
    'kmeans': KMeansData,
    'pca': PCAData
}

DATASET_LOADERS = {
    'iris': IrisData,
    'fish': FishData
}
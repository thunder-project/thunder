"""
Utilities for generating test datasets
"""

from numpy import array, random, arange, ceil, floor, dot, shape


class DataSets(object):

    def __init__(self, sc):
        self.sc = sc

    @staticmethod
    def create(sc, name, **opts):
        return DATASET_MAKERS[name](sc).make(**opts)

    def load(self):
        pass


class KMeansData(DataSets):

    def make(self, k, npartitions=10, ndims=5, nrecords=100, noise=0.1):
        centers = random.randn(k, ndims)
        gen_func = lambda i: (i, centers[int(floor(random.rand(1, 1) * k))] + noise*random.rand(ndims))
        data_local = map(gen_func, range(0, nrecords))
        self.centers = centers
        self.rdd = self.sc.parallelize(data_local, npartitions)
        return self


class PCAData(DataSets):

    def make(self, k, npartitions=10, nrows=100, ncols=10):
        U = random.randn(nrows, k)
        V = random.randn(k, ncols)
        X = dot(U, V)
        X += random.randn(shape(X)[0], shape(X)[1])
        data_local = map(lambda i: (i, X[i]), range(0, nrows))
        self.u = U
        self.v = V
        self.rdd = self.sc.parallelize(data_local, npartitions)
        return self



DATASET_MAKERS = {
    'kmeans': KMeansData,
    'pca': PCAData
}
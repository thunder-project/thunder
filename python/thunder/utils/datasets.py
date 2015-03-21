"""
Utilities for generating example datasets
"""

from numpy import array, asarray, random, shape, floor, dot, linspace, \
    sin, sign, c_, ceil, inf, clip, zeros

from thunder.rdds.matrices import RowMatrix
from thunder.rdds.series import Series


class DataSets(object):

    def __init__(self, sc, returnParams=False):
        self.sc = sc
        self.returnParams = returnParams

    @staticmethod
    def make(sc, name, returnParams=False, **opts):
        try:
            return DATASET_MAKERS[name.lower()](sc, returnParams).generate(**opts)
        except KeyError:
            raise NotImplementedError("no dataset generator for '%s'" % name)


# eliminate this
def appendKeys(data):

    data = array(data)
    n = shape(data)[0]
    x = (random.rand(n) * n).astype(int)
    y = (random.rand(n) * n).astype(int)
    z = (random.rand(n) * n).astype(int)
    dataZipped = zip(x, y, z, data)
    return map(lambda (k1, k2, k3, v): ((k1, k2, k3), v), dataZipped)


class KMeansData(DataSets):

    def generate(self, k=5, npartitions=10, ndims=5, nrecords=100, noise=0.1, seed=None):
        random.seed(seed)
        centers = random.randn(k, ndims)
        genFunc = lambda i: centers[int(floor(random.rand(1, 1) * k))] + noise*random.rand(ndims)
        dataLocal = map(genFunc, range(0, nrecords))
        data = Series(self.sc.parallelize(appendKeys(dataLocal), npartitions))
        if self.returnParams is True:
            return data, centers
        else:
            return data


class PCAData(DataSets):

    def generate(self, k=3, npartitions=10, nrows=100, ncols=10, seed=None):
        random.seed(seed)
        u = random.randn(nrows, k)
        v = random.randn(k, ncols)
        a = dot(u, v)
        a += random.randn(shape(a)[0], shape(a)[1])
        data = RowMatrix(self.sc.parallelize(appendKeys(a), npartitions))
        if self.returnParams is True:
            return data, u, v
        else:
            return data


class ICAData(DataSets):

    def generate(self, npartitions=10, nrows=100):
        random.seed(42)
        time = linspace(0, 10, nrows)
        s1 = sin(2 * time)
        s2 = sign(sin(3 * time))
        s = c_[s1, s2]
        s += 0.2 * random.randn(s.shape[0], s.shape[1])  # Add noise
        s /= s.std(axis=0)
        a = array([[1, 1], [0.5, 2]])
        x = dot(s, a.T)
        data = RowMatrix(self.sc.parallelize(appendKeys(x), npartitions))
        if self.returnParams is True:
            return data, s, a
        else:
            return data


class SourcesData(DataSets):

    def generate(self, x=100, y=100, n=5, t=100, margin=35, sd=3, npartitions=1, seed=None):

        from scipy.ndimage.filters import gaussian_filter
        from thunder.rdds.fileio.imagesloader import ImagesLoader

        random.seed(seed)
        ts = [clip(random.randn(t), 0, inf) for i in range(0, n)]
        xcenters = (x - margin) * random.random_sample(n) + margin/2
        ycenters = (y - margin) * random.random_sample(n) + margin/2
        centers = zip(xcenters, ycenters)
        allframes = []
        for tt in range(0, t):
            frame = zeros((x, y))
            for nn in range(0, n):
                base = zeros((x, y))
                base[centers[nn][0], centers[nn][1]] = 1
                img = gaussian_filter(base, sd)
                frame += img * ts[nn][tt]
            allframes.append(frame)

        data = ImagesLoader(self.sc).fromArrays(allframes, npartitions).astype('float')
        if self.returnParams is True:
            return data, ts, centers
        else:
            return data



DATASET_MAKERS = {
    'kmeans': KMeansData,
    'pca': PCAData,
    'ica': ICAData,
    'sources': SourcesData
}

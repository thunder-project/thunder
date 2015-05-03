"""
Utilities for generating example datasets
"""

from numpy import array, asarray, random, shape, floor, dot, linspace, \
    sin, sign, c_, ceil, inf, clip, zeros, max, size, sqrt, log, matrix

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

    @staticmethod
    def appendKeys(data):
        data = array(data)
        n = shape(data)[0]
        x = (random.rand(n) * n).astype(int)
        return zip(x, data)


class KMeansData(DataSets):

    def generate(self, k=5, npartitions=10, ndims=5, nrecords=100, noise=0.1, seed=None):
        random.seed(seed)
        centers = random.randn(k, ndims)
        genFunc = lambda i: centers[int(floor(random.rand(1, 1) * k))] + noise*random.rand(ndims)
        dataLocal = map(genFunc, range(0, nrecords))
        data = Series(self.sc.parallelize(self.appendKeys(dataLocal), npartitions))
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
        data = RowMatrix(self.sc.parallelize(self.appendKeys(a), npartitions))
        if self.returnParams is True:
            return data, u, v
        else:
            return data

class FactorAnalysisData(DataSets):

    def generate(self, q=1, p=3, nrows=50, npartitions=10, sigmas=None, seed=None):
        """
        Generate data from a factor analysis model

        Parameters
        ----------
        q : int, optional, default = 1
          The number of factors generating this data

        p : int, optios, default = 3
          The number of observed factors (p >= q)

        nrows : int, optional, default = 50
          Number of observations we have

        sigmas = 1 x p ndarray, optional, default = None
          Scale of the noise to add, randomly generated
          from standard normal distribution if not given
        """
        random.seed(seed)
        # Generate factor loadings (n x q)
        F = matrix(random.randn(nrows, q))
        # Generate factor scores (q x p)
        w = matrix(random.randn(q, p))
        # Generate non-zero the error covariances (1 x p)
        if sigmas is None:
          sigmas = random.randn(1, p)
        # Generate the error terms (n x p)
        # (each row gets scaled by our sigmas)
        epsilon = random.randn(nrows, p) * sigmas
        # Combine this to get our actual data (n x p)
        x = (F * w) + epsilon
        # Put the data in an RDD
        data = RowMatrix(self.sc.parallelize(self.appendKeys(x), npartitions))

        if self.returnParams is True:
            return data, F, w, epsilon
        else:
            return data

class RandomData(DataSets):

    def generate(self, nrows=50, ncols=50, npartitions=10, seed=None):
        """
        Generate a matrix where every element is i.i.d. and drawn from a
        standard normal distribution

        Parameters
        ----------
        nrows : int, optional, default = 50
          Number of columns in the generated matrix

        nrows : int, optional, default = 50
          Number of rows in the generated matrix
        """
        random.seed(seed)
        # Generate the data
        x = matrix(random.randn(nrows, ncols))
        # Put the data into an RDD
        data = RowMatrix(self.sc.parallelize(self.appendKeys(x), npartitions))
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
        data = RowMatrix(self.sc.parallelize(self.appendKeys(x), npartitions))
        if self.returnParams is True:
            return data, s, a
        else:
            return data


class SourcesData(DataSets):

    def generate(self, dims=(100, 200), centers=5, t=100, margin=35, sd=3, noise=0.1, npartitions=1, seed=None):

        from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
        from skimage.draw import circle
        from thunder.rdds.fileio.imagesloader import ImagesLoader
        from thunder.extraction.source import SourceModel

        random.seed(seed)

        if len(dims) != 2:
            raise Exception("Can only generate for two-dimensional sources.")

        if size(centers) == 1:
            n = centers
            xcenters = (dims[1] - margin) * random.random_sample(n) + margin/2
            ycenters = (dims[0] - margin) * random.random_sample(n) + margin/2
            centers = zip(xcenters, ycenters)
        else:
            centers = asarray(centers)
            n = len(centers)

        ts = [clip(random.randn(t), 0, inf) for i in range(0, n)]
        ts = [gaussian_filter1d(vec, 10) for vec in ts] * 5
        allframes = []
        for tt in range(0, t):
            frame = zeros(dims)
            for nn in range(0, n):
                base = zeros(dims)
                base[centers[nn][1], centers[nn][0]] = 1
                img = gaussian_filter(base, sd)
                img = img/max(img)
                frame += img * ts[nn][tt]
            frame += random.randn(dims[0], dims[1]) * noise
            allframes.append(frame)

        def pointToCircle(center, radius):
            rr, cc = circle(center[0], center[1], radius)
            return array(zip(rr, cc))

        r = round(sd * 1.5)
        sources = SourceModel([pointToCircle(c[::-1], r) for c in centers])

        data = ImagesLoader(self.sc).fromArrays(allframes, npartitions).astype('float')
        if self.returnParams is True:
            return data, ts, sources
        else:
            return data


DATASET_MAKERS = {
    'kmeans': KMeansData,
    'pca': PCAData,
    'factor': FactorAnalysisData,
    'rand': RandomData,
    'ica': ICAData,
    'sources': SourcesData
}

"""
Classes for KMeans clustering
"""

from numpy import array, argmin, corrcoef, ndarray, asarray, std, random, floor

from ..data.series.series import Series
from ..data.series.readers import fromlist


class KMeansModel(object):
    """
    Estimated KMeans model and its parameters.

    Parameters
    ----------
    centers : array
        Cluster centers

    Attributes
    ----------
    centers : array
        Cluster centers

    colors : array
        Unique color labels for each cluster
    """

    def __init__(self, centers):

        self.centers = centers

    def calc(self, data, func):
        """Base function for making clustering predictions"""

        # small optimization to avoid serializing full model
        centers = self.centers

        if isinstance(data, Series):
            return data.apply_values(lambda x: func(centers, x))

        elif isinstance(data, list):
            return map(lambda x: func(centers, x), data)

        elif isinstance(data, ndarray):
            if data.ndim == 1:
                return func(centers, data)
            else:
                return map(lambda x: func(centers, x), data)

    def predict(self, data):
        """
        Predict the cluster that all data points belong to.

        Parameters
        ----------
        data : Series or subclass (e.g. RowMatrix), a list of arrays, or a single array
            The data to predict cluster assignments on

        Returns
        -------
        closest : Series, list of arrays, or a single array
            For each data point, ggives the closest center to that point
        """
        from scipy.spatial.distance import cdist
        closest = lambda centers, p: argmin(cdist(centers, array([p])))
        out = self.calc(data, closest)
        if isinstance(data, Series):
            out._index = 'label'
        return out

    def similarity(self, data):
        """
        Estimate similarity between each data point and the cluster it belongs to.

        Parameters
        ----------
        data : Series or subclass (e.g. RowMatrix), a list of arrays, or a single array
            The data to estimate similarities on

        Returns
        -------
        similarities : Series, list of arrays, or a single array
            For each data point, gives the similarity to its nearest cluster
        """
        from scipy.spatial.distance import cdist
        similarity = lambda centers, p: 0 if std(p) == 0 else \
            corrcoef(centers[argmin(cdist(centers, array([p])))], p)[0, 1]
        out = self.calc(data, similarity)
        if isinstance(data, Series):
            out._index = 'similarity'
        return out


class KMeans(object):
    """
    KMeans clustering algorithm.

    Parameters
    ----------
    k : int
        Number of clusters to find

    maxiter : int, optional, default = 20
        Maximum number of iterations to use

    tol : float, optional, default = 0.001
        Change tolerance for stopping algorithm
    """
    def __init__(self, k, maxIterations=20):
        self.k = k
        self.maxIterations = maxIterations

    def fit(self, data):
        """
        Train the clustering model using the implementation
        of KMeans from mllib.

        Parameters
        ----------
        data :  Series or a subclass (e.g. RowMatrix)
            The data to cluster

        Returns
        -------
        centers : KMeansModel
            The estimated cluster centers
        """

        if not (isinstance(data, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        import pyspark.mllib.clustering as mllib

        model = mllib.KMeans.train(data.astype('float').rdd.values(), k=self.k, maxIterations=self.maxIterations)

        return KMeansModel(asarray(model.clusterCenters))

    @staticmethod
    def make(shape=(100, 5), k=5, noise=0.1, npartitions=10, seed=None, withparams=False):
        """
        Generator random data for testing clustering
        """
        random.seed(seed)
        centers = random.randn(k, shape[1])
        gen = lambda i: centers[int(floor(random.rand(1, 1) * k))] + noise*random.rand(shape[1])
        local = map(gen, range(0, shape[0]))
        data = fromlist(local, npartitions=npartitions)
        if withparams is True:
            return data, centers
        else:
            return data
from numpy import random, dot

from .svd import SVD
from ..data.series.series import Series
from ..data.series.matrix import Matrix
from ..data.series.readers import fromList

class PCA(object):
    """
    Principal components analysis on a distributed matrix.

    Parameters
    ----------
    k : int
        Number of principal components to estimate

    svdMethod : str, optional, default = "auto"
        If set to 'direct', will compute the SVD with direct gramian matrix estimation and eigenvector decomposition.
        If set to 'em', will approximate the SVD using iterative expectation-maximization algorithm.
        If set to 'auto', will use 'em' if number of columns in input data exceeds 750, otherwise will use 'direct'.

    Attributes
    ----------
    `comps` : array, shape (k, ncols)
        The k principal components

    `latent` : array, shape (k,)
        The latent values

    `scores` : RowMatrix, nrows, each of shape (k,)
        The scores (i.e. the representation of the data in PC space)

    See also
    --------
    SVD : singular value decomposition
    """

    def __init__(self, k=3, svdMethod='auto'):
        self.k = k
        self.svdMethod = svdMethod
        self.scores = None
        self.latent = None
        self.comps = None

    def fit(self, data):
        """
        Estimate principal components

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays
        """

        if not (isinstance(data, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if type(data) is not Matrix:
            data = data.tomatrix()

        mat = data.center(1)

        svd = SVD(k=self.k, method=self.svdMethod)
        svd.calc(mat)

        self.scores = svd.u
        self.latent = svd.s
        self.comps = svd.v

        return self

    def transform(self, data):
        """
        Project data into principal component space

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        Returns
        -------
        scores : RowMatrix, nrows, each of shape (k,)
            The scores (i.e. the representation of the data in PC space)
        """

        if not (isinstance(data, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if type(data) is not Matrix:
            data = data.tomatrix()

        mat = data.center(1)
        scores = mat.times(self.comps.T / self.latent)
        return scores

    @staticmethod
    def make(shape=(100, 10), k=3, npartitions=10, seed=None, withparams=False):
        """
        Generator random data for PCA
        """
        random.seed(seed)
        u = random.randn(shape[0], k)
        v = random.randn(k, shape[1])
        a = dot(u, v)
        a += random.randn(a.shape[0], a.shape[1])
        data = fromList(a, npartitions=npartitions)
        if withparams is True:
            return data, u, v
        else:
            return data
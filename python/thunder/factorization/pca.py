"""
Class for Principal Component Analysis
"""

from thunder.factorization import SVD
from thunder.rdds import Series, RowMatrix


class PCA(object):
    """
    Principal components analysis on a distributed matrix.

    Parameters
    ----------
    k : int
        Number of principal components to estimate

    svdmethod : str, optional, default = "direct"
        Which method to use for performing the SVD

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

    def __init__(self, k=3, svdmethod='direct'):
        self.k = k
        self.svdmethod = svdmethod

    def fit(self, data):
        """Estimate principal components

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays
        """

        if not (isinstance(data, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if type(data) is not RowMatrix:
            data = data.toRowMatrix()

        mat = data.center(0)

        svd = SVD(k=self.k, method=self.svdmethod)
        svd.calc(mat)

        self.scores = svd.u
        self.latent = svd.s
        self.comps = svd.v

        return self

    def transform(self, data):
        """Project data into principal component space

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

        if type(data) is not RowMatrix:
            data = RowMatrix(data)

        mat = data.center(0)
        scores = mat.times(self.comps.T / self.latent)
        return scores

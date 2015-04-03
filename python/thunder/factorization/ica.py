"""
Class for Independent Component Analysis
"""

from thunder.factorization.svd import SVD
from thunder.rdds.series import Series
from thunder.rdds.matrices import RowMatrix


class ICA(object):
    """
    Independent component analysis on a distributed matrix.

    Initial dimensionality reduction performed via SVD

    Parameters
    ----------
    k : int
        Number of principal components to use

    c : int
        Number of independent components to estimate

    svdMethod : string, optional, default = "auto"
        Which SVD method to use. If set to 'auto',
        will select preferred method based on dimensionality.

    maxIter : Int, optional, default = 10
        Maximum number of iterations

    tol : float, optional, default = 0.00001
        Tolerance for convergence

    Attributes
    ----------
    `w` : array, shape (c, ncols)
        Recovered unmixing matrix

    `a` : array, shape (ncols, ncols)
        Recovered mixing matrix

    `sigs` : RowMatrix, nrows, each array of shape (c,)
        Estimated independent components

    """

    def __init__(self, c, k=None, svdMethod='auto', maxIter=10, tol=0.000001, seed=0):
        self.k = k
        self.c = c
        self.svdMethod = svdMethod
        self.maxIter = maxIter
        self.tol = tol
        self.seed = seed
        self.w = None
        self.a = None
        self.sigs = None

    def fit(self, data):
        """
        Fit independent components using an iterative fixed-point algorithm

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        Returns
        ----------
        self : returns an instance of self.
        """

        from numpy import random, sqrt, zeros, real, dot, outer, diag, transpose
        from scipy.linalg import sqrtm, inv, orth

        if not (isinstance(data, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if not isinstance(data, RowMatrix):
            data = data.toRowMatrix()

        d = data.ncols

        if self.k is None:
            self.k = d

        if self.c > self.k:
            raise Exception("number of independent comps " + str(self.c) +
                            " must be less than the number of principal comps " + str(self.k))

        if self.k > d:
            raise Exception("number of principal comps " + str(self.k) +
                            " must be less than the data dimensionality " + str(d))

        # reduce dimensionality
        svd = SVD(k=self.k, method=self.svdMethod).calc(data)

        # whiten data
        whtMat = real(dot(inv(diag(svd.s/sqrt(data.nrows))), svd.v))
        unWhtMat = real(dot(transpose(svd.v), diag(svd.s/sqrt(data.nrows))))
        wht = data.times(whtMat.T)

        # do multiple independent component extraction
        if self.seed != 0:
            random.seed(self.seed)
        b = orth(random.randn(self.k, self.c))
        bOld = zeros((self.k, self.c))
        niter = 0
        minAbsCos = 0
        errVec = zeros(self.maxIter)

        while (niter < self.maxIter) & ((1 - minAbsCos) > self.tol):
            niter += 1
            # update rule for pow3 non-linearity (TODO: add others)
            b = wht.rows().map(lambda x: outer(x, dot(x, b) ** 3)).sum() / wht.nrows - 3 * b
            # make orthogonal
            b = dot(b, real(sqrtm(inv(dot(transpose(b), b)))))
            # evaluate error
            minAbsCos = min(abs(diag(dot(transpose(b), bOld))))
            # store results
            bOld = b
            errVec[niter-1] = (1 - minAbsCos)

        # get un-mixing matrix
        w = dot(b.T, whtMat)

        # get mixing matrix
        a = dot(unWhtMat, b)

        # get components
        sigs = data.times(w.T)

        self.w = w
        self.a = a
        self.sigs = sigs

        return self

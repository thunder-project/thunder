"""
Class for Independent Component Analysis
"""

from numpy import random, sqrt, zeros, real, dot, outer, diag, transpose
from scipy.linalg import sqrtm, inv, orth
from thunder.factorization import SVD
from thunder.rdds import Series, RowMatrix


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

    svdmethod : string, optional, default = "direct"
        Which SVD method to use

    maxiter : Int, optional, default = 10
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

    def __init__(self, c, k=None, svdmethod="direct", maxiter=10, tol=0.000001, seed=0):
        self.k = k
        self.c = c
        self.svdmethod = svdmethod
        self.maxiter = maxiter
        self.tol = tol
        self.seed = seed

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
        svd = SVD(k=self.k, method=self.svdmethod).calc(data)

        # whiten data
        whtmat = real(dot(inv(diag(svd.s/sqrt(data.nrows))), svd.v))
        unwhtmat = real(dot(transpose(svd.v), diag(svd.s/sqrt(data.nrows))))
        wht = data.times(whtmat.T)

        # do multiple independent component extraction
        if self.seed != 0:
            random.seed(self.seed)
        b = orth(random.randn(self.k, self.c))
        b_old = zeros((self.k, self.c))
        iter = 0
        minabscos = 0
        errvec = zeros(self.maxiter)

        while (iter < self.maxiter) & ((1 - minabscos) > self.tol):
            iter += 1
            # update rule for pow3 non-linearity (TODO: add others)
            b = wht.rows().map(lambda x: outer(x, dot(x, b) ** 3)).sum() / wht.nrows - 3 * b
            # make orthogonal
            b = dot(b, real(sqrtm(inv(dot(transpose(b), b)))))
            # evaluate error
            minabscos = min(abs(diag(dot(transpose(b), b_old))))
            # store results
            b_old = b
            errvec[iter-1] = (1 - minabscos)

        # get un-mixing matrix
        w = dot(b.T, whtmat)

        # get mixing matrix
        a = dot(unwhtmat, b)

        # get components
        sigs = data.times(w.T)

        self.w = w
        self.a = a
        self.sigs = sigs

        return self

from numpy import sign, random, linspace, sin, array, dot, c_

from .svd import SVD
from ..data.series.series import Series
from ..data.series.matrix import Matrix
from ..data.series.readers import fromlist


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
        Which SVD method to use.
        If set to 'direct', will compute the SVD with direct gramian matrix estimation and eigenvector decomposition.
        If set to 'em', will approximate the SVD using iterative expectation-maximization algorithm.
        If set to 'auto', will use 'em' if number of columns in input data exceeds 750, otherwise will use 'direct'.

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

    See also
    --------
    SVD : singular value decomposition
    PCA: principal components analysis
    """

    def __init__(self, c, k=None, svdmethod='auto', maxiterations=10, tol=0.000001, seed=0):
        self.k = k
        self.c = c
        self.svdmethod = svdmethod
        self.maxiterations = maxiterations
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

        if not isinstance(data, Matrix):
            data = data.tomatrix()

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
        bold = zeros((self.k, self.c))
        niter = 0
        minabscos = 0
        errVec = zeros(self.maxiterations)

        while (niter < self.maxiterations) & ((1 - minabscos) > self.tol):
            niter += 1
            # update rule for pow3 non-linearity (TODO: add others)
            b = wht.rows().map(lambda x: outer(x, dot(x, b) ** 3)).sum() / wht.nrows - 3 * b
            # make orthogonal
            b = dot(b, real(sqrtm(inv(dot(transpose(b), b)))))
            # evaluate error
            minAbsCos = min(abs(diag(dot(transpose(b), bold))))
            # store results
            bold = b
            errVec[niter-1] = (1 - minAbsCos)

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

    @staticmethod
    def make(shape=(100, 10), npartitions=10, withparams=False):
        """
        Generator random data for ICA
        """
        random.seed(42)
        time = linspace(0, shape[1], shape[0])
        s1 = sin(2 * time)
        s2 = sign(sin(3 * time))
        s = c_[s1, s2]
        s += 0.2 * random.randn(s.shape[0], s.shape[1])  # Add noise
        s /= s.std(axis=0)
        a = array([[1, 1], [0.5, 2]])
        x = dot(s, a.T)
        data = fromlist(x, npartitions=npartitions)
        if withparams is True:
            return data, s, a
        else:
            return data

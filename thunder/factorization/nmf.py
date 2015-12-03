from numpy import add, any, diag, dot, inf, maximum, outer, sqrt, apply_along_axis
from numpy.linalg import inv, norm, pinv
from numpy.random import rand

from ..data.series.series import Series


class NMF(object):
    """
    Non-negative matrix factorization on a distributed matrix.

    Parameters
    ----------
    method : string, optional, default='als'
        Specifies which iterative algorithm is to be used.

    k : int, optional, default=5
        Size of low-dimensional basis.

    maxIter : int, optional, default=20
        Maximum number of iterations.

    tol : float, optional, default=0.001
        Tolerance for convergence of iterative algorithm.

    h0 : non-negative k x ncols array, optional
        Value at which H is initialized.

    history : boolean, optional, default=False
        Whether to compute reconstruction at each step.

    verbose : boolean, optional, default=False
        Whether to print progress.

    Attributes
    ----------
    `w` : Distributed array of nrows (tuple, array) pairs, each array of shape (k,)
        Left bases.

    `h` : array, shape (k, ncols)
        Right bases.

    'convergence` : list of floats
        List of Frobenius norms between successive estimates of h.

    `error` : list of floats
        Output of the reconstruction error at each iteration.
    """

    def __init__(self, k=5, method='als', maxiterations=20, tolerance=0.001, h0=None,
                 history=False, verbose=False):
        self.k = int(k)
        self.method = method
        self.maxiterations = maxiterations
        self.tolerance = tolerance
        self.history = history
        self.verbose = verbose
        self.h0 = h0
        self.h = None
        self.w = None
        self.convergence = list()

        if self.history:
            self.error = list()

    def fit(self, mat):
        """
        Calcuate the non-negative matrix decomposition.

        Parameters
        ----------
        mat : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        Returns
        ----------
        self : returns an instance of self.
        """

        # TODO use RowMatrix throughout

        if not (isinstance(mat, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        mat = mat.rdd

        def frobenius(a, b):
            return sqrt(a.zip(b).map(lambda ((akey, x), (bkey, y)): sum((x - y) ** 2)).reduce(add))

        k = self.k
        if k < 1:
            raise ValueError("Provided k must be greater than 1.")

        m = mat.values().first().size
        if self.h0 is not None:
            if any(self.h0 < 0):
                raise ValueError("Provided h0 contains negative entries.")

        if self.method == "als":

            iteration = 0
            convergence = 100

            if self.h0 is None:
                self.h0 = rand(k, m)

            h = self.h0
            w = None

            # goal is to solve R = WH subject to all entries of W,H >= 0
            # by iteratively updating W and H with least squares and clipping negative values
            while (iteration < self.maxiterations) and (convergence > self.tolerance):
                # update values on iteration
                h_old = h

                # precompute pinv(H) = inv(H' x H) * H' (easy here because h is an np array)
                pinvH = pinv(h)

                # update W using least squares with R * pinv(H); then clip negative values to 0
                w = mat.mapValues(lambda x: dot(x, pinvH))

                # clip negative values of W
                # noinspection PyUnresolvedReferences
                w = w.mapValues(lambda x: maximum(x, 0))

                # precompute inv(W' * W) to get inv_gramian_w, a np array
                # We have chosen k to be small, i.e., rank(W) = k, so W'*W is invertible
                gramianW = w.values().map(lambda x: outer(x, x)).reduce(add)
                invGramianW = inv(gramianW)

                # pseudoinverse of W is inv(W' * W) * W' = inv_gramian_w * w
                pinvW = w.mapValues(lambda x: dot(invGramianW, x))

                # update H using least squares with inv(W' * W) * W * R (same as pinv(W) * R)
                h = pinvW.values().zip(mat.values()).map(lambda (x, y): outer(x, y)).reduce(add)

                # clip negative values of H
                # noinspection PyUnresolvedReferences
                h = maximum(h, 0)

                # normalize the rows of H
                # noinspection PyUnresolvedReferences
                h = dot(diag(1 / maximum(apply_along_axis(norm, 1, h), 0.001)), h)

                # estimate convergence
                convergence = norm(h-h_old)
                self.convergence.append(convergence)

                # calculate reconstruction error
                if self.history:
                    datarecon = w.mapValues(lambda x: dot(x, h))
                    self.error.append(frobenius(mat, datarecon))

                # report progress
                if self.verbose:
                    print "Finished als iteration %d with " \
                          "convergence = %.6f in H" % (iteration, convergence)

                # increment count
                iteration += 1

            # report convergence
            if self.verbose:
                if convergence <= self.tolerance:
                    print "Converged to specified tolerance."
                else:
                    print "Warning: reached maxiter without converging to specified tolerance."

            # calculate reconstruction error
            if self.history:
                datarecon = w.mapValues(lambda x: dot(x, h))
                self.error.append(frobenius(mat, datarecon))

            # TODO: need to propagate metadata through to this new Series object
            self.h = h
            self.w = Series(w)

        else:
            raise Exception("Algorithm %s is not supported" % self.method)

        return self

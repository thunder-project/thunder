import shutil
import tempfile
from numpy import array, allclose, transpose, random, dot, corrcoef, diag
from numpy.linalg import norm
import scipy.linalg as LinAlg
from thunder.factorization.ica import ICA
from thunder.factorization.svd import SVD
from thunder.factorization.nmf import NMF
from thunder.utils.datasets import DataSets
from thunder.rdds.matrices import RowMatrix
from test_utils import PySparkTestCase


class FactorizationTestCase(PySparkTestCase):
    def setUp(self):
        super(FactorizationTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(FactorizationTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestSVD(FactorizationTestCase):
    """Test accuracy of direct and em methods
    for SVD against scipy.linalg method,

    Only uses k=1 otherwise results of iterative approaches can
    vary up to an orthogonal transform

    Checks if answers match up to a sign flip
    """
    def test_svd_direct(self):
        data_local = [
            array([1.0, 2.0, 6.0]),
            array([1.0, 3.0, 0.0]),
            array([1.0, 4.0, 6.0]),
            array([5.0, 1.0, 4.0])
        ]
        data = self.sc.parallelize(zip(range(1, 5), data_local))
        mat = RowMatrix(data)

        svd = SVD(k=1, method="direct")
        svd.calc(mat)
        u_true, s_true, v_true = LinAlg.svd(array(data_local))
        u_test = transpose(array(svd.u.rows().collect()))[0]
        v_test = svd.v[0]
        assert(allclose(svd.s[0], s_true[0]))
        assert(allclose(v_test, v_true[0, :]) | allclose(-v_test, v_true[0, :]))
        assert(allclose(u_test, u_true[:, 0]) | allclose(-u_test, u_true[:, 0]))

    def test_svd_em(self):
        data_local = [
            array([1.0, 2.0, 6.0]),
            array([1.0, 3.0, 0.0]),
            array([1.0, 4.0, 6.0]),
            array([5.0, 1.0, 4.0])
        ]
        data = self.sc.parallelize(zip(range(1, 5), data_local))
        mat = RowMatrix(data)

        svd = SVD(k=1, method="em")
        svd.calc(mat)
        u_true, s_true, v_true = LinAlg.svd(array(data_local))
        u_test = transpose(array(svd.u.rows().collect()))[0]
        v_test = svd.v[0]
        tol = 10e-04  # allow small error for iterative method
        assert(allclose(svd.s[0], s_true[0], atol=tol))
        assert(allclose(v_test, v_true[0, :], atol=tol) | allclose(-v_test, v_true[0, :], atol=tol))
        assert(allclose(u_test, u_true[:, 0], atol=tol) | allclose(-u_test, u_true[:, 0], atol=tol))

    def test_conversion(self):
        from thunder.rdds.series import Series
        data_local = [
            array([1.0, 2.0, 6.0]),
            array([1.0, 3.0, 0.0]),
            array([1.0, 4.0, 6.0]),
            array([5.0, 1.0, 4.0])
        ]
        data = Series(self.sc.parallelize(zip(range(1, 5), data_local)))
        SVD(k=1, method='direct').calc(data)


class TestICA(FactorizationTestCase):
    """Test ICA results against ground truth,
    taking into account possible sign flips and permutations
    """
    def test_ica(self):

        random.seed(42)
        data, s, a = DataSets.make(self.sc, "ica", nrows=100, returnParams=True)

        ica = ICA(c=2, svdMethod="direct", seed=1)
        ica.fit(data)

        s_ = array(ica.sigs.rows().collect())

        # test accurate recovery of original signals
        tol = 0.01
        assert(allclose(abs(corrcoef(s[:, 0], s_[:, 0])[0, 1]), 1, atol=tol)
               or allclose(abs(corrcoef(s[:, 0], s_[:, 1])[0, 1]), 1, atol=tol))
        assert(allclose(abs(corrcoef(s[:, 1], s_[:, 0])[0, 1]), 1, atol=tol)
               or allclose(abs(corrcoef(s[:, 1], s_[:, 1])[0, 1]), 1, atol=tol))

        # test accurate reconstruction from sources
        assert(allclose(array(data.rows().collect()), dot(s_, ica.a.T)))


class TestNMF(FactorizationTestCase):
    def test_als(self):
        """ Test accuracy of alternating least-squares NMF algorithm
        against the MATLAB-computed version
        """
        #  set data and initializing constants
        keys = [array([i+1]) for i in range(4)]
        data_local = array([
            [1.0, 2.0, 6.0],
            [1.0, 3.0, 0.0],
            [1.0, 4.0, 6.0],
            [5.0, 1.0, 4.0]])
        data = self.sc.parallelize(zip(keys, data_local))
        mat = RowMatrix(data)
        h0 = array(
            [[0.09082617,  0.85490047,  0.57234593],
             [0.82766740,  0.21301186,  0.90913979]])

        # if the rows of h are not normalized on each iteration:
        h_true = array(
            [[0.    ,    0.6010,    0.9163],
             [0.8970,    0.1556,    0.7423]])
        w_true = array(
            [[4.5885,    1.5348],
             [1.3651,    0.2184],
             [5.9349,    1.0030],
             [0.    ,    5.5147]])

        # if the columns of h are normalized (as in the current implementation):
        scale_mat = diag(norm(h_true, axis=1))
        h_true = dot(LinAlg.inv(scale_mat), h_true)
        w_true = dot(w_true, scale_mat)

        # calculate NMF using the Thunder implementation
        # (maxiter=9 corresponds with Matlab algorithm)
        nmf_thunder = NMF(k=2, method="als", h0=h0, maxIter=9)
        nmf_thunder.fit(mat)
        h_thunder = nmf_thunder.h
        w_thunder = array(nmf_thunder.w.values().collect())

        tol = 1e-03  # allow small error
        assert(allclose(w_thunder, w_true, atol=tol))
        assert(allclose(h_thunder, h_true, atol=tol))

    def test_init(self):
        """
        test performance of whole function, including random initialization
        """
        data_local = array([
            [1.0, 2.0, 6.0],
            [1.0, 3.0, 0.0],
            [1.0, 4.0, 6.0],
            [5.0, 1.0, 4.0]])
        data = self.sc.parallelize(zip([array([i]) for i in range(data_local.shape[0])], data_local))
        mat = RowMatrix(data)

        nmf_thunder = NMF(k=2, reconHist='final')
        nmf_thunder.fit(mat)

        # check to see if Thunder's solution achieves close-to-optimal reconstruction error
        # scikit-learn's solution achieves 2.993952
        # matlab's non-deterministic implementation usually achieves < 2.9950 (when it converges)
        assert(nmf_thunder.reconErr < 2.9950)
import os
import shutil
import tempfile
from numpy import array, allclose, transpose
import scipy.linalg as LinAlg
from scipy.io import loadmat
from thunder.factorization import ICA
from thunder.factorization import SVD
from thunder.io import load
from test_utils import PySparkTestCase

# Hack to find the data files:
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")


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

        svd = SVD(k=1, method="direct")
        svd.calc(data)
        u_true, s_true, v_true = LinAlg.svd(array(data_local))
        u_test = transpose(array(svd.u.map(lambda (_, v): v).collect()))[0]
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

        svd = SVD(k=1, method="em")
        svd.calc(data)
        u_true, s_true, v_true = LinAlg.svd(array(data_local))
        u_test = transpose(array(svd.u.map(lambda (_, v): v).collect()))[0]
        v_test = svd.v[0]
        tol = 10e-04  # allow small error for iterative method
        assert(allclose(svd.s[0], s_true[0], atol=tol))
        assert(allclose(v_test, v_true[0, :], atol=tol) | allclose(-v_test, v_true[0, :], atol=tol))
        assert(allclose(u_test, u_true[:, 0], atol=tol) | allclose(-u_test, u_true[:, 0], atol=tol))


class TestICA(FactorizationTestCase):
    """Test that ICA returns correct
    results by comparing to known, vetted
    results for the example data set
    and a fixed random seed
    """
    def test_ica(self):
        ica_data = os.path.join(DATA_DIR, "ica.txt")
        ica_results = os.path.join(DATA_DIR, "results/ica")
        data = load(self.sc, ica_data, "raw")
        ica = ICA(k=4, c=4, svdmethod="direct", seed=1)
        ica.fit(data)
        w_true = loadmat(os.path.join(ica_results, "w.mat"))["w"]
        sigs_true = loadmat(os.path.join(ica_results, "sigs.mat"))["sigs"]
        tol = 10e-02
        assert(allclose(ica.w, w_true, atol=tol))
        assert(allclose(transpose(ica.sigs.map(lambda (_, v): v).collect()), sigs_true, atol=tol))


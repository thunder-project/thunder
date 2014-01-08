import os
import shutil
import tempfile
from numpy import array, allclose, transpose
import scipy.linalg as LinAlg
from scipy.io import loadmat
from thunder.factorization.ica import ica
from thunder.factorization.util import svd
from thunder.util.dataio import parse
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
    """test accuracy of direct and em methods
    for SVD against scipy.linalg method,
    only use k=1 otherwise results of em can
    vary up to an orthogonal transform
    """
    def test_svd_direct(self):
        data_local = array([
            array([1.0, 2.0, 6.0]),
            array([1.0, 3.0, 0.0]),
            array([1.0, 4.0, 6.0]),
            array([5.0, 1.0, 4.0])
        ])
        data = self.sc.parallelize(data_local)

        u, s, v = svd(data, 1, meansubtract=0, method="direct")
        u_true, s_true, v_true = LinAlg.svd(data_local)
        u_test = transpose(array(u.collect()))[0]
        v_test = v[0]
        assert(allclose(s[0], s_true[0]))
        assert(allclose(v_test, v_true[0, :]) | allclose(-v_test, v_true[0, :]))
        assert(allclose(u_test, u_true[:, 0]) | allclose(-u_test, u_true[:, 0]))

    def test_svd_em(self):
        data_local = array([
            array([1.0, 2.0, 6.0]),
            array([1.0, 3.0, 0.0]),
            array([1.0, 4.0, 6.0]),
            array([5.0, 1.0, 4.0])
        ])
        data = self.sc.parallelize(data_local)

        u, s, v = svd(data, 1, meansubtract=0, method="em")
        u_true, s_true, v_true = LinAlg.svd(data_local)
        u_test = transpose(array(u.collect()))[0]
        v_test = v[0]
        tol = 10e-04  # allow for small error for iterative method
        assert(allclose(s[0], s_true[0], atol=tol))
        assert(allclose(v_test, v_true[0, :], atol=tol) | allclose(-v_test, v_true[0, :], atol=tol))
        assert(allclose(u_test, u_true[:, 0], atol=tol) | allclose(-u_test, u_true[:, 0], atol=tol))


class TestICA(FactorizationTestCase):
    """test that ICA returns correct
    results by comparing to known, vetted
    results for the example data set
    and a fixed random seed
    """
    def test_ica(self):
        ica_data = os.path.join(DATA_DIR, "ica.txt")
        ica_results = os.path.join(DATA_DIR, "results/ica")
        data = parse(self.sc.textFile(ica_data), "raw")
        w, sigs = ica(data, 4, 4, svdmethod="direct", seed=1)
        w_true = loadmat(os.path.join(ica_results, "w.mat"))["w"]
        sigs_true = loadmat(os.path.join(ica_results, "sigs.mat"))["sigs"]
        tol = 10e-02
        assert(allclose(w, w_true, atol=tol))
        assert(allclose(transpose(sigs.collect()), sigs_true, atol=tol))


import shutil
import tempfile
from numpy import array, allclose, transpose, random, dot, corrcoef
import scipy.linalg as LinAlg
from thunder.factorization import ICA
from thunder.factorization import SVD
from thunder.utils import DataSets
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
    """Test ICA results against ground truth,
    taking into account possible sign flips and permutations
    """
    def test_ica(self):

        random.seed(42)
        data, s, a = DataSets.make(self.sc, "ica", nrows=100, params=True)

        ica = ICA(c=2, svdmethod="direct", seed=1)
        ica.fit(data)

        s_ = array(ica.sigs.values().collect())

        # test accurate recovery of original signals
        tol = 0.01
        assert(allclose(abs(corrcoef(s[:, 0], s_[:, 0])[0, 1]), 1, atol=tol)
               or allclose(abs(corrcoef(s[:, 0], s_[:, 1])[0, 1]), 1, atol=tol))
        assert(allclose(abs(corrcoef(s[:, 1], s_[:, 0])[0, 1]), 1, atol=tol)
               or allclose(abs(corrcoef(s[:, 1], s_[:, 1])[0, 1]), 1, atol=tol))

        # test accurate reconstruction from sources
        assert(allclose(array(data.values().collect()), dot(s_, ica.a.T)))

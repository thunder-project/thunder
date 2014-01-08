import os
import shutil
import tempfile
from numpy import array, allclose, mean, median, std, corrcoef
from scipy.linalg import norm
from thunder.sigprocessing.util import SigProcessingMethod
from thunder.sigprocessing.stats import stats
from thunder.sigprocessing.fourier import fourier
from thunder.sigprocessing.crosscorr import crosscorr
from thunder.sigprocessing.localcorr import localcorr
from thunder.sigprocessing.query import query
from test_utils import PySparkTestCase


class SigProcessingTestCase(PySparkTestCase):
    def setUp(self):
        super(SigProcessingTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(SigProcessingTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestStats(SigProcessingTestCase):
    """test accuracy for signal statistics
    by comparison to direct evaluation using numpy/scipy
    """
    def test_stats(self):
        data_local = array([
            array([1.0, 2.0, -4.0, 5.0]),
            array([2.0, 2.0, -4.0, 5.0]),
            array([3.0, 2.0, -4.0, 5.0]),
            array([4.0, 2.0, -4.0, 5.0]),
        ])

        data = self.sc.parallelize(data_local)

        vals = stats(data, "mean")
        print(vals.collect())
        assert(allclose(vals.collect(), mean(data_local, axis=1)))

        vals = stats(data, "median")
        assert(allclose(vals.collect(), median(data_local, axis=1)))

        vals = stats(data, "std")
        assert(allclose(vals.collect(), std(data_local, axis=1)))

        vals = stats(data, "norm")
        for i in range(0, 4):
            assert(allclose(vals.collect()[i], norm(data_local[i, :] - mean(data_local[i, :]))))


class TestFourier(SigProcessingTestCase):
    """test accuracy for fourier analysis
    by comparison to known result
    (verified in MATLAB)
    """
    def test_fourier(self):
        data_local = array([
            array([1.0, 2.0, -4.0, 5.0, 8.0, 3.0, 4.1, 0.9, 2.3]),
            array([2.0, 2.0, -4.0, 5.0, 3.1, 4.5, 8.2, 8.1, 9.1]),
        ])

        data = self.sc.parallelize(data_local)

        co, ph = fourier(data, 2)
        assert(allclose(co.collect()[0], 0.578664))
        assert(allclose(ph.collect()[0], 4.102501))


class TestLocalCorr(SigProcessingTestCase):
    """test accuracy for local correlation
    by comparison to known result
    (verified by directly computing
    result with numpy's mean and corrcoef)

    try with indexing from both 0 and 1
    """
    def test_fourier_0_indexing(self):

        data_local = [
            ((0, 0, 0), array([1.0, 2.0, 3.0])),
            ((0, 1, 0), array([2.0, 2.0, 4.0])),
            ((0, 2, 0), array([9.0, 2.0, 1.0])),
            ((1, 0, 0), array([5.0, 2.0, 5.0])),
            ((2, 0, 0), array([4.0, 2.0, 6.0])),
            ((1, 1, 0), array([4.0, 2.0, 8.0])),
            ((1, 2, 0), array([5.0, 4.0, 1.0])),
            ((2, 1, 0), array([6.0, 3.0, 2.0])),
            ((2, 2, 0), array([0.0, 2.0, 1.0]))
        ]

        # get ground truth by correlating mean with the center
        ts = map(lambda x: x[1], data_local)
        mn = mean(ts, axis=0)
        truth = corrcoef(mn, array([4.0, 2.0, 8.0]))[0, 1]

        data = self.sc.parallelize(data_local)

        corr = localcorr(data, 1)
        assert(allclose(corr.collect()[4], truth))

    def test_fourier_1_indexing(self):

        data_local = [
            ((1, 1, 1), array([1.0, 2.0, 3.0])),
            ((1, 2, 1), array([2.0, 2.0, 4.0])),
            ((1, 3, 1), array([9.0, 2.0, 1.0])),
            ((2, 1, 1), array([5.0, 2.0, 5.0])),
            ((3, 1, 1), array([4.0, 2.0, 6.0])),
            ((2, 2, 1), array([4.0, 2.0, 8.0])),
            ((2, 3, 1), array([5.0, 4.0, 1.0])),
            ((3, 2, 1), array([6.0, 3.0, 2.0])),
            ((3, 3, 1), array([0.0, 2.0, 1.0]))
        ]

        # get ground truth by correlating mean with the center
        ts = map(lambda x: x[1], data_local)
        mn = mean(ts, axis=0)
        truth = corrcoef(mn, array([4.0, 2.0, 8.0]))[0, 1]

        data = self.sc.parallelize(data_local)

        corr = localcorr(data, 1)
        assert(allclose(corr.collect()[4], truth))


class TestQuery(SigProcessingTestCase):
    """test accuracy for query
    by comparison to known result
    (calculated by hand)
    """
    def test_query(self):
        data_local = [
            (1, array([1.0, 2.0, 3.0])),
            (2, array([2.0, 2.0, 4.0])),
            (3, array([4.0, 2.0, 1.0]))
        ]

        data = self.sc.parallelize(data_local)

        inds = array([array([1, 2]), array([3])])

        ts = query(data, inds)
        assert(allclose(ts[0, :], array([1.5, 2., 3.5])))
        assert(allclose(ts[1, :], array([4.0, 2.0, 1.0])))


class TestCrossCorr(SigProcessingTestCase):
    """test accuracy for cross correlation
    by comparison to known result
    (lag=0 case tested with numpy corrcoef function,
    lag>0 case tested against result from MATLAB's xcov)

    also tests main analysis script
    """
    def test_crosscorr(self):
        data_local = array([
            array([1.0, 2.0, -4.0, 5.0, 8.0, 3.0, 4.1, 0.9, 2.3]),
            array([2.0, 2.0, -4.0, 5.0, 3.1, 4.5, 8.2, 8.1, 9.1]),
        ])

        sig = array([1.5, 2.1, -4.2, 5.6, 8.1, 3.9, 4.2, 0.3, 2.1])

        data = self.sc.parallelize(data_local)

        method = SigProcessingMethod.load("crosscorr", sigfile=sig, lag=0)
        betas = method.calc(data)
        assert(allclose(betas.collect()[0], corrcoef(data_local[0, :], sig)[0, 1]))
        assert(allclose(betas.collect()[1], corrcoef(data_local[1, :], sig)[0, 1]))

        method = SigProcessingMethod.load("crosscorr", sigfile=sig, lag=2)
        betas = method.calc(data)
        tol = 1E-5  # to handle rounding errors
        assert(allclose(betas.collect()[0], array([-0.18511, 0.03817, 0.99221, 0.06567, -0.25750]), atol=tol))
        assert(allclose(betas.collect()[1], array([-0.35119, -0.14190, 0.44777, -0.00408, 0.45435]), atol=tol))

        betas = crosscorr(data, sig, 0)
        assert(allclose(betas.collect()[0], corrcoef(data_local[0, :], sig)[0, 1]))
        assert(allclose(betas.collect()[1], corrcoef(data_local[1, :], sig)[0, 1]))







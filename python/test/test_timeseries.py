import shutil
import tempfile
from numpy import array, allclose, corrcoef
from thunder.rdds.timeseries import TimeSeries
from test_utils import PySparkTestCase
from nose.tools import assert_equals


class TimeSeriesTestCase(PySparkTestCase):
    def setUp(self):
        super(TimeSeriesTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(TimeSeriesTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestTimeSeriesMethods(TimeSeriesTestCase):

    def test_fourier(self):
        dataLocal = [
            array([1.0, 2.0, -4.0, 5.0, 8.0, 3.0, 4.1, 0.9, 2.3]),
            array([2.0, 2.0, -4.0, 5.0, 3.1, 4.5, 8.2, 8.1, 9.1]),
        ]

        rdd = self.sc.parallelize(zip(range(1, 3), dataLocal))
        data = TimeSeries(rdd)
        vals = data.fourier(freq=2)

        assert(allclose(vals.select('coherence').values().collect()[0], 0.578664))
        assert(allclose(vals.select('phase').values().collect()[0], 4.102501))

    def test_convolve(self):
        dataLocal = array([1, 2, 3, 4, 5])
        sig = array([1, 2, 3])
        rdd = self.sc.parallelize([(0, dataLocal)])
        data = TimeSeries(rdd)
        betas = data.convolve(sig, mode='same')
        assert(allclose(betas.values().collect()[0], array([4, 10, 16, 22, 22])))

    def test_crossCorr(self):
        dataLocal = array([
            array([1.0, 2.0, -4.0, 5.0, 8.0, 3.0, 4.1, 0.9, 2.3]),
            array([2.0, 2.0, -4.0, 5.0, 3.1, 4.5, 8.2, 8.1, 9.1]),
        ])

        sig = array([1.5, 2.1, -4.2, 5.6, 8.1, 3.9, 4.2, 0.3, 2.1])

        rdd = self.sc.parallelize(zip(range(1, 3), dataLocal))
        data = TimeSeries(rdd)

        betas = data.crossCorr(signal=sig, lag=0)

        assert(allclose(betas.values().collect()[0], corrcoef(dataLocal[0, :], sig)[0, 1]))
        assert(allclose(betas.values().collect()[1], corrcoef(dataLocal[1, :], sig)[0, 1]))

        betas = data.crossCorr(signal=sig, lag=2)
        tol = 1E-5  # to handle rounding errors
        assert(allclose(betas.values().collect()[0], array([-0.18511, 0.03817, 0.99221, 0.06567, -0.25750]), atol=tol))
        assert(allclose(betas.values().collect()[1], array([-0.35119, -0.14190, 0.44777, -0.00408, 0.45435]), atol=tol))

    def test_detrend(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = TimeSeries(rdd).detrend('linear')
        # detrending linearly increasing data should yield all 0s
        assert(allclose(data.first()[1], array([0, 0, 0, 0, 0])))

    def test_normalization_bypercentile(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5], dtype='float16'))])
        data = TimeSeries(rdd, dtype='float16')
        out = data.normalize('percentile', perc=20)
        # check that _dtype has been set properly *before* calling first(), b/c first() will update this
        # value even if it hasn't been correctly set
        assert_equals('float16', str(out._dtype))
        vals = out.first()[1]
        assert_equals('float16', str(vals.dtype))
        assert(allclose(vals, array([-0.42105,  0.10526,  0.63157,  1.15789,  1.68421]), atol=1e-3))

    def test_normalization_bywindow(self):
        y = array([1, 2, 3, 4, 5], dtype='float16')
        rdd = self.sc.parallelize([(0, y)])
        data = TimeSeries(rdd, dtype='float16')
        out = data.normalize('window', window=2)
        # check that _dtype has been set properly *before* calling first(), b/c first() will update this
        # value even if it hasn't been correctly set
        assert_equals('float16', str(out._dtype))
        vals = out.first()[1]
        assert_equals('float64', str(vals.dtype))
        b_true = array([1.2,  1.4,  2.4,  3.4,  4.2])
        result_true = (y - b_true) / (b_true + 0.1)
        assert(allclose(vals, result_true, atol=1e-3))

        out = data.normalize('window', window=6)
        vals = out.first()[1]
        b_true = array([1.6,  1.8,  1.8,  1.8,  2.6])
        result_true = (y - b_true) / (b_true + 0.1)
        assert(allclose(vals, result_true, atol=1e-3))

    def test_normalization_bymean(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5], dtype='float16'))])
        data = TimeSeries(rdd, dtype='float16')
        out = data.normalize('mean')
        # check that _dtype has been set properly *before* calling first(), b/c first() will update this
        # value even if it hasn't been correctly set
        assert_equals('float16', str(out._dtype))
        vals = out.first()[1]
        assert_equals('float16', str(vals.dtype))
        assert(allclose(out.first()[1],
                        array([-0.64516,  -0.32258,  0.0,  0.32258,  0.64516]), atol=1e-3))

    # TODO add test for triggered average

    # TODO add test for blocked averaged


import shutil
import tempfile
from numpy import array, allclose, corrcoef
from thunder.rdds.timeseries import TimeSeries
from test_utils import PySparkTestCase


class TimeSeriesTestCase(PySparkTestCase):
    def setUp(self):
        super(TimeSeriesTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(TimeSeriesTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestTimeSeriesMethods(TimeSeriesTestCase):

    def fourier_test(self):
        data_local = [
            array([1.0, 2.0, -4.0, 5.0, 8.0, 3.0, 4.1, 0.9, 2.3]),
            array([2.0, 2.0, -4.0, 5.0, 3.1, 4.5, 8.2, 8.1, 9.1]),
        ]

        rdd = self.sc.parallelize(zip(range(1, 3), data_local))
        data = TimeSeries(rdd)
        vals = data.fourier(freq=2)

        assert(allclose(vals.select('coherence').values().collect()[0], 0.578664))
        assert(allclose(vals.select('phase').values().collect()[0], 4.102501))

    def cross_corr_test(self):
        data_local = array([
            array([1.0, 2.0, -4.0, 5.0, 8.0, 3.0, 4.1, 0.9, 2.3]),
            array([2.0, 2.0, -4.0, 5.0, 3.1, 4.5, 8.2, 8.1, 9.1]),
        ])

        sig = array([1.5, 2.1, -4.2, 5.6, 8.1, 3.9, 4.2, 0.3, 2.1])

        rdd = self.sc.parallelize(zip(range(1, 3), data_local))
        data = TimeSeries(rdd)

        betas = data.crossCorr(signal=sig, lag=0)

        assert(allclose(betas.values().collect()[0], corrcoef(data_local[0, :], sig)[0, 1]))
        assert(allclose(betas.values().collect()[1], corrcoef(data_local[1, :], sig)[0, 1]))

        betas = data.crossCorr(signal=sig, lag=2)
        tol = 1E-5  # to handle rounding errors
        assert(allclose(betas.values().collect()[0], array([-0.18511, 0.03817, 0.99221, 0.06567, -0.25750]), atol=tol))
        assert(allclose(betas.values().collect()[1], array([-0.35119, -0.14190, 0.44777, -0.00408, 0.45435]), atol=tol))

    # TODO add test for triggered average

    # TODO add test for blocked averaged


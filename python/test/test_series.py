from numpy import allclose, amax, arange, array, array_equal, dtype
from nose.tools import assert_equals, assert_true

from thunder.rdds.series import Series
from test_utils import *


class TestSeriesConversions(PySparkTestCase):

    def test_to_row_matrix(self):
        from thunder.rdds.matrices import RowMatrix
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd)
        mat = data.toRowMatrix()
        assert(isinstance(mat, RowMatrix))
        assert(mat.nrows == 2)
        assert(mat.ncols == 4)

    def test_to_time_series(self):
        from thunder.rdds.timeseries import TimeSeries
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd)
        ts = data.toTimeSeries()
        assert(isinstance(ts, TimeSeries))

    def test_cast_to_float(self):
        from numpy import arange
        shape = (3, 2, 2)
        size = 3*2*2
        ary = arange(size, dtype='uint8').reshape(shape)
        ary2 = ary + size
        from thunder.rdds.fileio.seriesloader import SeriesLoader
        series = SeriesLoader(self.sc).fromArrays([ary, ary2])

        castseries = series.astype("smallfloat")

        assert_equals('float16', str(castseries.dtype))
        assert_equals('float16', str(castseries.first()[1].dtype))


class TestSeriesDataStatsMethods(PySparkTestCase):
    def generateTestSeries(self):
        from thunder.rdds.fileio.seriesloader import SeriesLoader
        ary1 = arange(8, dtype=dtype('uint8')).reshape((2, 4))
        ary2 = arange(8, 16, dtype=dtype('uint8')).reshape((2, 4))
        return SeriesLoader(self.sc).fromArrays([ary1, ary2])

    def test_mean(self):
        from test_utils import elementwise_mean
        series = self.generateTestSeries()
        meanval = series.mean()

        expected = elementwise_mean(series.values().collect())
        assert_true(allclose(expected, meanval))
        assert_equals('float16', str(meanval.dtype))

    def test_sum(self):
        from numpy import add
        series = self.generateTestSeries()
        sumval = series.sum(dtype='float32')

        arys = series.values().collect()
        expected = reduce(add, arys)
        assert_true(array_equal(expected, sumval))
        assert_equals('float32', str(sumval.dtype))

    def test_variance(self):
        from test_utils import elementwise_var
        series = self.generateTestSeries()
        varval = series.variance()

        arys = series.values().collect()
        expected = elementwise_var([ary.astype('float16') for ary in arys])
        assert_true(allclose(expected, varval))
        assert_equals('float16', str(varval.dtype))

    def test_stdev(self):
        from test_utils import elementwise_stdev
        series = self.generateTestSeries()
        stdval = series.stdev()

        arys = series.values().collect()
        expected = elementwise_stdev([ary.astype('float16') for ary in arys])
        assert_true(allclose(expected, stdval, atol=0.001))
        assert_equals('float32', str(stdval.dtype))  # why not float16? see equivalent Images test

    def test_stats(self):
        from test_utils import elementwise_mean, elementwise_var
        series = self.generateTestSeries()
        statsval = series.stats()

        arys = series.values().collect()
        floatarys = [ary.astype('float16') for ary in arys]
        expectedmean = elementwise_mean(floatarys)
        expectedvar = elementwise_var(floatarys)
        assert_true(allclose(expectedmean, statsval.mean()))
        assert_true(allclose(expectedvar, statsval.variance()))

    def test_max(self):
        from numpy import maximum
        series = self.generateTestSeries()
        maxval = series.max()
        arys = series.values().collect()
        assert_true(array_equal(reduce(maximum, arys), maxval))

    def test_min(self):
        from numpy import minimum
        series = self.generateTestSeries()
        minval = series.min()
        arys = series.values().collect()
        assert_true(array_equal(reduce(minimum, arys), minval))


class TestSeriesMethods(PySparkTestCase):

    def test_between(self):
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd).between(0, 1)
        assert(allclose(data.index, array([0, 1])))
        assert(allclose(data.first()[1], array([4, 5])))

    def test_select(self):
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd, index=['label1', 'label2', 'label3', 'label4'])
        selection1 = data.select(['label1'])
        assert(allclose(selection1.first()[1], 4))
        selection1 = data.select('label1')
        assert(allclose(selection1.first()[1], 4))
        selection2 = data.select(['label1', 'label2'])
        assert(allclose(selection2.first()[1], array([4, 5])))

    def test_series_stats(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd)
        assert(allclose(data.seriesMean().first()[1], 3.0))
        assert(allclose(data.seriesSum().first()[1], 15.0))
        assert(allclose(data.seriesMedian().first()[1], 3.0))
        assert(allclose(data.seriesStdev().first()[1], 1.4142135))
        assert(allclose(data.seriesStat('mean').first()[1], 3.0))
        assert(allclose(data.seriesStats().select('mean').first()[1], 3.0))
        assert(allclose(data.seriesStats().select('count').first()[1], 5))
        assert(allclose(data.seriesPercentile(25).first()[1], 2.0))
        assert(allclose(data.seriesPercentile((25, 75)).first()[1], array([2.0, 4.0])))

    def test_standardization_axis0(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5], dtype='float16'))])
        data = Series(rdd, dtype='float16')
        centered = data.center(0)
        standardized = data.standardize(0)
        zscored = data.zscore(0)
        assert_equals('float16', centered._dtype)
        assert_equals('float16', standardized._dtype)
        assert_equals('float16', zscored._dtype)
        assert(allclose(centered.first()[1], array([-2, -1, 0, 1, 2]), atol=1e-3))
        assert(allclose(standardized.first()[1], array([0.70710,  1.41421,  2.12132,  2.82842,  3.53553]), atol=1e-3))
        assert(allclose(zscored.first()[1], array([-1.41421, -0.70710,  0,  0.70710,  1.41421]), atol=1e-3))

    def test_standardization_axis1(self):
        rdd = self.sc.parallelize([(0, array([1, 2], dtype='float16')), (0, array([3, 4], dtype='float16'))])
        data = Series(rdd, dtype='float16')
        centered = data.center(1)
        standardized = data.standardize(1)
        zscored = data.zscore(1)
        assert_equals('float16', centered._dtype)
        assert_equals('float16', standardized._dtype)
        assert_equals('float16', zscored._dtype)
        assert(allclose(centered.first()[1], array([-1, -1]), atol=1e-3))
        assert(allclose(standardized.first()[1], array([1, 2]), atol=1e-3))
        assert(allclose(zscored.first()[1], array([-1, -1]), atol=1e-3))

    def test_correlate(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5], dtype='float16'))])
        data = Series(rdd, dtype='float16')
        sig1 = [4, 5, 6, 7, 8]
        corrdata = data.correlate(sig1)
        assert_equals('float64', corrdata._dtype)
        corr = corrdata.values().collect()
        assert(allclose(corr[0], 1))
        sig12 = [[4, 5, 6, 7, 8], [8, 7, 6, 5, 4]]
        corrs = data.correlate(sig12).values().collect()
        assert(allclose(corrs[0], [1, -1]))

    def test_query_subscripts(self):
        data_local = [
            ((1, 1), array([1.0, 2.0, 3.0])),
            ((2, 1), array([2.0, 2.0, 4.0])),
            ((1, 2), array([4.0, 2.0, 1.0]))
        ]

        data = Series(self.sc.parallelize(data_local))

        inds = array([array([1, 2]), array([3])])
        keys, values = data.query(inds)
        assert(allclose(values[0, :], array([1.5, 2., 3.5])))
        assert(allclose(values[1, :], array([4.0, 2.0, 1.0])))

    def test_query_linear(self):
        data_local = [
            ((1,), array([1.0, 2.0, 3.0])),
            ((2,), array([2.0, 2.0, 4.0])),
            ((3,), array([4.0, 2.0, 1.0]))
        ]

        data = Series(self.sc.parallelize(data_local))

        inds = array([array([1, 2]), array([3])])
        keys, values = data.query(inds)
        assert(allclose(values[0, :], array([1.5, 2., 3.5])))
        assert(allclose(values[1, :], array([4.0, 2.0, 1.0])))

    def test_query_linear_singleton(self):
        data_local = [
            ((1,), array([1.0, 2.0, 3.0])),
            ((2,), array([2.0, 2.0, 4.0])),
            ((3,), array([4.0, 2.0, 1.0]))
        ]

        data = Series(self.sc.parallelize(data_local))

        inds = array([array([1, 2])])
        keys, values = data.query(inds)
        assert(allclose(values[0, :], array([1.5, 2., 3.5])))
        assert_equals(data.dtype, values[0, :].dtype)

    def test_maxProject(self):
        from thunder.rdds.fileio.seriesloader import SeriesLoader
        ary = arange(8, dtype=dtype('int16')).reshape((2, 4))

        series = SeriesLoader(self.sc).fromArrays(ary)
        project0Series = series.maxProject(axis=0)
        project0 = project0Series.pack()

        project1Series = series.maxProject(axis=1)
        project1 = project1Series.pack(sorting=True)

        assert_true(array_equal(amax(ary.T, 0), project0))
        assert_true(array_equal(amax(ary.T, 1), project1))

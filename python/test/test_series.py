from numpy import allclose, amax, arange, array, array_equal
from numpy import dtype as dtypeFunc
from nose.tools import assert_equals, assert_true

from thunder.rdds.series import Series
from test_utils import *


class TestSeriesConversions(PySparkTestCase):

    def test_toRowMatrix(self):
        from thunder.rdds.matrices import RowMatrix
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd)
        mat = data.toRowMatrix()
        assert(isinstance(mat, RowMatrix))
        assert(mat.nrows == 2)
        assert(mat.ncols == 4)

    def test_toTimeSeries(self):
        from thunder.rdds.timeseries import TimeSeries
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd)
        ts = data.toTimeSeries()
        assert(isinstance(ts, TimeSeries))

    def test_castToFloat(self):
        from numpy import arange
        shape = (3, 2, 2)
        size = 3*2*2
        ary = arange(size, dtype=dtypeFunc('uint8')).reshape(shape)
        ary2 = ary + size
        from thunder.rdds.fileio.seriesloader import SeriesLoader
        series = SeriesLoader(self.sc).fromArrays([ary, ary2])

        castSeries = series.astype("smallfloat")

        assert_equals('float16', str(castSeries.dtype))
        assert_equals('float16', str(castSeries.first()[1].dtype))


class TestSeriesDataStatsMethods(PySparkTestCase):
    def generateTestSeries(self):
        from thunder.rdds.fileio.seriesloader import SeriesLoader
        ary1 = arange(8, dtype=dtypeFunc('uint8')).reshape((2, 4))
        ary2 = arange(8, 16, dtype=dtypeFunc('uint8')).reshape((2, 4))
        return SeriesLoader(self.sc).fromArrays([ary1, ary2])

    def test_mean(self):
        from test_utils import elementwiseMean
        series = self.generateTestSeries()
        meanVal = series.mean()

        expected = elementwiseMean(series.values().collect())
        assert_true(allclose(expected, meanVal))
        assert_equals('float64', str(meanVal.dtype))

    def test_sum(self):
        from numpy import add
        series = self.generateTestSeries()
        sumVal = series.sum(dtype='float32')

        arys = series.values().collect()
        expected = reduce(add, arys)
        assert_true(array_equal(expected, sumVal))
        assert_equals('float32', str(sumVal.dtype))

    def test_variance(self):
        from test_utils import elementwiseVar
        series = self.generateTestSeries()
        varVal = series.variance()

        arys = series.values().collect()
        expected = elementwiseVar([ary.astype('float16') for ary in arys])
        assert_true(allclose(expected, varVal))
        assert_equals('float64', str(varVal.dtype))

    def test_stdev(self):
        from test_utils import elementwiseStdev
        series = self.generateTestSeries()
        stdVal = series.stdev()

        arys = series.values().collect()
        expected = elementwiseStdev([ary.astype('float16') for ary in arys])
        assert_true(allclose(expected, stdVal, atol=0.001))
        assert_equals('float64', str(stdVal.dtype))  # why not float16? see equivalent Images test

    def test_stats(self):
        from test_utils import elementwiseMean, elementwiseVar
        series = self.generateTestSeries()
        statsVal = series.stats()

        arys = series.values().collect()
        floatArys = [ary.astype('float16') for ary in arys]
        expectedMean = elementwiseMean(floatArys)
        expectedVar = elementwiseVar(floatArys)
        assert_true(allclose(expectedMean, statsVal.mean()))
        assert_true(allclose(expectedVar, statsVal.variance()))

    def test_max(self):
        from numpy import maximum
        series = self.generateTestSeries()
        maxVal = series.max()
        arys = series.values().collect()
        assert_true(array_equal(reduce(maximum, arys), maxVal))

    def test_min(self):
        from numpy import minimum
        series = self.generateTestSeries()
        minVal = series.min()
        arys = series.values().collect()
        assert_true(array_equal(reduce(minimum, arys), minVal))


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

    def test_seriesStats(self):
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
        corrData = data.correlate(sig1)
        assert_equals('float64', corrData._dtype)
        corr = corrData.values().collect()
        assert(allclose(corr[0], 1))
        sig12 = [[4, 5, 6, 7, 8], [8, 7, 6, 5, 4]]
        corrs = data.correlate(sig12).values().collect()
        assert(allclose(corrs[0], [1, -1]))

    def test_query_subscripts(self):
        dataLocal = [
            ((1, 1), array([1.0, 2.0, 3.0])),
            ((2, 1), array([2.0, 2.0, 4.0])),
            ((1, 2), array([4.0, 2.0, 1.0]))
        ]

        data = Series(self.sc.parallelize(dataLocal))

        inds = array([array([1, 2]), array([3])])
        keys, values = data.query(inds)
        assert(allclose(values[0, :], array([1.5, 2., 3.5])))
        assert(allclose(values[1, :], array([4.0, 2.0, 1.0])))

    def test_query_linear(self):
        dataLocal = [
            ((1,), array([1.0, 2.0, 3.0])),
            ((2,), array([2.0, 2.0, 4.0])),
            ((3,), array([4.0, 2.0, 1.0]))
        ]

        data = Series(self.sc.parallelize(dataLocal))

        inds = array([array([1, 2]), array([3])])
        keys, values = data.query(inds)
        assert(allclose(values[0, :], array([1.5, 2., 3.5])))
        assert(allclose(values[1, :], array([4.0, 2.0, 1.0])))

    def test_query_linear_singleton(self):
        dataLocal = [
            ((1,), array([1.0, 2.0, 3.0])),
            ((2,), array([2.0, 2.0, 4.0])),
            ((3,), array([4.0, 2.0, 1.0]))
        ]

        data = Series(self.sc.parallelize(dataLocal))

        inds = array([array([1, 2])])
        keys, values = data.query(inds)
        assert(allclose(values[0, :], array([1.5, 2., 3.5])))
        assert_equals(data.dtype, values[0, :].dtype)

    def __setup_meanByRegion(self):
        dataLocal = [
            ((0, 0), array([1.0, 2.0, 3.0])),
            ((0, 1), array([2.0, 2.0, 4.0])),
            ((1, 0), array([4.0, 2.0, 1.0])),
            ((1, 1), array([3.0, 1.0, 1.0]))
        ]
        series = Series(self.sc.parallelize(dataLocal))
        itemIdxs = [1, 2]  # data keys for items 1 and 2 (0-based)
        keys = [dataLocal[idx][0] for idx in itemIdxs]

        expectedKeys = tuple(vstack([dataLocal[idx][0] for idx in itemIdxs]).mean(axis=0).astype('int16'))
        expected = vstack([dataLocal[idx][1] for idx in itemIdxs]).mean(axis=0)
        return series, keys, expectedKeys, expected

    def test_meanByRegion(self):
        series, keys, expectedKeys, expected = self.__setup_meanByRegion()

        actual = series.meanByRegion(keys)
        assert_equals(2, len(actual))
        assert_equals(expectedKeys, actual[0])
        assert_true(array_equal(expected, actual[1]))

    def test_meanByRegions_singleRegion(self):
        series, keys, expectedKeys, expected = self.__setup_meanByRegion()

        actualSeries = series.meanByRegions([keys])
        actual = actualSeries.collect()
        assert_equals(1, len(actual))
        assert_equals(expectedKeys, actual[0][0])
        assert_true(array_equal(expected, actual[0][1]))

    def test_maxProject(self):
        from thunder.rdds.fileio.seriesloader import SeriesLoader
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))

        series = SeriesLoader(self.sc).fromArrays(ary)
        project0Series = series.maxProject(axis=0)
        project0 = project0Series.pack()

        project1Series = series.maxProject(axis=1)
        project1 = project1Series.pack(sorting=True)

        assert_true(array_equal(amax(ary.T, 0), project0))
        assert_true(array_equal(amax(ary.T, 1), project1))

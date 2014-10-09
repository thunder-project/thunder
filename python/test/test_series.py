from numpy import array, allclose

from thunder.rdds.series import Series
from test_utils import PySparkTestCase


class TestConversions(PySparkTestCase):

    def test_to_row_matrix(self):
        from thunder.rdds import RowMatrix
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd)
        mat = data.toRowMatrix()
        assert(isinstance(mat, RowMatrix))
        assert(mat.nrows == 2)
        assert(mat.ncols == 4)

    def test_to_time_series(self):
        from thunder.rdds import TimeSeries
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd)
        ts = data.toTimeSeries()
        assert(isinstance(ts, TimeSeries))


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

    def test_detrend(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd).detrend('linear')
        # detrending linearly increasing data should yield all 0s
        assert(allclose(data.first()[1], array([0, 0, 0, 0, 0])))

    def test_series_stats(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd)
        assert(allclose(data.seriesMean().first()[1], 3.0))
        assert(allclose(data.seriesSum().first()[1], 15.0))
        assert(allclose(data.seriesStdev().first()[1], 1.4142135))
        assert(allclose(data.seriesStat('mean').first()[1], 3.0))
        assert(allclose(data.seriesStats().select('mean').first()[1], 3.0))
        assert(allclose(data.seriesStats().select('count').first()[1], 5))

    def test_normalization(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd)
        assert(allclose(data.normalize('percentile').first()[1],
                        array([-0.42105,  0.10526,  0.63157,  1.15789,  1.68421]), atol=1e-4))

    def test_standardization_axis0(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd)
        assert(allclose(data.center(0).first()[1], array([-2, -1, 0, 1, 2])))
        assert(allclose(data.standardize(0).first()[1], array([0.70710,  1.41421,  2.12132,  2.82842,  3.53553])))
        assert(allclose(data.zscore(0).first()[1], array([-1.41421, -0.70710,  0,  0.70710,  1.41421])))

    def test_standardization_axis1(self):
        rdd = self.sc.parallelize([(0, array([1, 2])), (0, array([3, 4]))])
        data = Series(rdd)
        assert(allclose(data.center(1).first()[1], array([-1, -1])))
        assert(allclose(data.standardize(1).first()[1], array([1, 2])))
        assert(allclose(data.zscore(1).first()[1], array([-1, -1])))

    def test_correlate(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd)
        sig1 = [4, 5, 6, 7, 8]
        corr = data.correlate(sig1).values().collect()
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
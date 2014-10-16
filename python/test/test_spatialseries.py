from numpy import array, allclose, mean, corrcoef
from test_utils import PySparkTestCase
from thunder.rdds.spatialseries import SpatialSeries


class TestLocalCorr(PySparkTestCase):
    """Test accuracy for local correlation
    by comparison to known result
    (verified by directly computing
    result with numpy's mean and corrcoef)

    Test with indexing from both 0 and 1
    """
    def test_localcorr_0_indexing(self):

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

        data = SpatialSeries(self.sc.parallelize(data_local))

        corr = data.localCorr(1)

        assert(allclose(corr.collect()[4][1], truth))

    def test_localcorr_1_indexing(self):

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

        data = SpatialSeries(self.sc.parallelize(data_local))

        corr = data.localCorr(1)

        assert(allclose(corr.collect()[4][1], truth))




import pytest
from numpy import allclose, arange, array

from thunder.data.series.readers import fromList
from thunder.data.series.matrix import Matrix
from thunder.data.series.timeseries import TimeSeries

pytestmark = pytest.mark.usefixtures("context")


def test_to_matrix():
    data = fromList([array([4, 5, 6, 7]), array([8, 9, 10, 11])])
    mat = data.toMatrix()
    assert isinstance(mat, Matrix)
    assert mat.nrows == 2
    assert mat.ncols == 4
    out = mat.applyvalues(lambda x: x + 1).toarray()
    assert allclose(out, array([[5, 6, 7, 8], [9, 10, 11, 12]]))


def test_to_time_series():
    data = fromList([array([4, 5, 6, 7]), array([8, 9, 10, 11])])
    ts = data.toTimeSeries()
    assert isinstance(ts, TimeSeries)


def test_between():
    data = fromList([array([4, 5, 6, 7]), array([8, 9, 10, 11])])
    val = data.between(0, 1)
    assert allclose(val.index, array([0, 1]))
    assert allclose(val.values().first(), array([4, 5]))


def test_select():
    index = ['label1', 'label2', 'label3', 'label4']
    data = fromList([array([4, 5, 6, 7]), array([8, 9, 10, 11])], index=index)
    assert allclose(data.select(['label1']).values().first(), 4)
    assert allclose(data.select(['label1', 'label2']).values().first(), array([4, 5]))


def test_series_stats():
    data = fromList([array([1, 2, 3, 4, 5])])
    assert allclose(data.seriesMean().values().first(), 3.0)
    assert allclose(data.seriesSum().values().first(), 15.0)
    assert allclose(data.seriesMedian().values().first(), 3.0)
    assert allclose(data.seriesStdev().values().first(), 1.4142135)
    assert allclose(data.seriesStat('mean').values().first(), 3.0)
    assert allclose(data.seriesStats().select('mean').values().first(), 3.0)
    assert allclose(data.seriesStats().select('count').values().first(), 5)
    assert allclose(data.seriesPercentile(25).values().first(), 2.0)
    assert allclose(data.seriesPercentile((25, 75)).values().first(), array([2.0, 4.0]))


def test_standardization_axis0():
    data = fromList([array([1, 2, 3, 4, 5])])
    centered = data.center(0)
    standardized = data.standardize(0)
    zscored = data.zscore(0)
    assert allclose(centered.values().first(), array([-2, -1, 0, 1, 2]), atol=1e-3)
    assert allclose(standardized.values().first(), 
        array([0.70710,  1.41421,  2.12132,  2.82842,  3.53553]), atol=1e-3)
    assert allclose(zscored.values().first(), 
        array([-1.41421, -0.70710,  0,  0.70710,  1.41421]), atol=1e-3)


def test_standardization_axis1():
    data = fromList([array([1, 2]), array([3, 4])])
    centered = data.center(1)
    standardized = data.standardize(1)
    zscored = data.zscore(1)
    assert allclose(centered.values().first(), array([-1, -1]), atol=1e-3)
    assert allclose(standardized.values().first(), array([1, 2]), atol=1e-3)
    assert allclose(zscored.values().first(), array([-1, -1]), atol=1e-3)


def test_squelch():
    data = fromList([array([1, 2]), array([3, 4])])
    squelched = data.squelch(5)
    assert allclose(squelched.toarray(), [[0, 0], [0, 0]])
    squelched = data.squelch(3)
    assert allclose(squelched.toarray(), [[0, 0], [3, 4]])
    squelched = data.squelch(1)
    assert allclose(squelched.toarray(), [[1, 2], [3, 4]])


def test_correlate():
    data = fromList([array([1, 2, 3, 4, 5])])
    sig = [4, 5, 6, 7, 8]
    corr = data.correlate(sig).values().first()
    assert allclose(corr, 1)
    sigs = [[4, 5, 6, 7, 8], [8, 7, 6, 5, 4]]
    corrs = data.correlate(sigs).values().first()
    assert allclose(corrs, [1, -1])


def test_subset():
    data = fromList([array([1, 5]), array([1, 10]), array([1, 15])])
    assert len(data.subset(3, stat='min', thresh=0)) == 3
    assert allclose(data.subset(1, stat='max', thresh=10), [[1, 15]])
    assert allclose(data.subset(1, stat='mean', thresh=6), [[1, 15]])
    assert allclose(data.subset(1, stat='std', thresh=6), [[1, 15]])
    assert allclose(data.subset(1, thresh=6), [[1, 15]])


def test_index_setting():
    data = fromList([array([1, 2, 3]), array([2, 2, 4]), array([4, 2, 1])])
    assert allclose(data.index, array([0, 1, 2]))
    data.index = [3, 2, 1]
    assert allclose(data.index, [3, 2, 1])
    with pytest.raises(ValueError):
        data.index = 5
    with pytest.raises(ValueError):
        data.index = [1, 2]


def test_select_by_index():
    data = fromList([arange(12)], index=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    result = data.selectByIndex(1)
    assert allclose(result.values().first(), array([4, 5, 6, 7]))
    assert allclose(result.index, array([1, 1, 1, 1]))
    result = data.selectByIndex(1, squeeze=True)
    assert allclose(result.index, array([0, 1, 2, 3]))
    index = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
    ]
    data.index = array(index).T
    result, mask = data.selectByIndex(0, level=2, returnMask=True)
    assert allclose(result.values().first(), array([0, 2, 6, 8]))
    assert allclose(result.index, array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]))
    assert allclose(mask, array([1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]))
    result = data.selectByIndex(0, level=2, squeeze=True)
    assert allclose(result.values().first(), array([0, 2, 6, 8]))
    assert allclose(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    result = data.selectByIndex([1, 0], level=[0, 1])
    assert allclose(result.values().first(), array([6, 7]))
    assert allclose(result.index, array([[1, 0, 0], [1, 0, 1]]))
    result = data.selectByIndex(val=[0, [2,3]], level=[0, 2])
    assert allclose(result.values().first(), array([4, 5]))
    assert allclose(result.index, array([[0, 1, 2], [0, 1, 3]]))
    result = data.selectByIndex(1, level=1, filter=True)
    assert allclose(result.values().first(), array([0, 1, 6, 7]))
    assert allclose(result.index, array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]))


def test_series_aggregate_by_index():
    data = fromList([arange(12)], index=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    result = data.seriesAggregateByIndex(sum)
    assert allclose(result.values().first(), array([6, 22, 38]))
    assert allclose(result.index, array([0, 1, 2]))
    index = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
    ]
    data.index = array(index).T
    result = data.seriesAggregateByIndex(sum, level=[0, 1])
    assert allclose(result.values().first(), array([1, 14, 13, 38]))
    assert allclose(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]]))


def test_series_stat_by_index():
    data = fromList([arange(12)], index=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    assert allclose(data.seriesStatByIndex('sum').values().first(), array([6, 22, 38]))
    assert allclose(data.seriesStatByIndex('mean').values().first(), array([1.5, 5.5, 9.5]))
    assert allclose(data.seriesStatByIndex('min').values().first(), array([0, 4, 8]))
    assert allclose(data.seriesStatByIndex('max').values().first(), array([3, 7, 11]))
    assert allclose(data.seriesStatByIndex('count').values().first(), array([4, 4, 4]))
    assert allclose(data.seriesStatByIndex('median').values().first(), array([1.5, 5.5, 9.5]))
    assert allclose(data.seriesSumByIndex().values().first(), array([6, 22, 38]))
    assert allclose(data.seriesMeanByIndex().values().first(), array([1.5, 5.5, 9.5]))
    assert allclose(data.seriesMinByIndex().values().first(), array([0, 4, 8]))
    assert allclose(data.seriesMaxByIndex().values().first(), array([3, 7, 11]))
    assert allclose(data.seriesCountByIndex().values().first(), array([4, 4, 4]))
    assert allclose(data.seriesMedianByIndex().values().first(), array([1.5, 5.5, 9.5]))


def test_series_stat_by_multiindex():
    index = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
    ]
    data = fromList([arange(12)], index=array(index).T)
    result = data.seriesStatByIndex('sum', level=[0, 1])
    assert allclose(result.values().first(), array([1, 14, 13, 38]))
    assert allclose(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    result = data.seriesSumByIndex(level=[0, 1])
    assert allclose(result.values().first(), array([1, 14, 13, 38]))
    assert allclose(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]]))


def test_group_by_fixed_length():
    data = fromList([arange(8)])
    test1 = data.groupByFixedLength(4)
    assert test1.keys().collect() == [(0, 0), (0, 1)]
    assert allclose(test1.index, array([0, 1, 2, 3]))
    assert allclose(test1.values().collect(), [[0, 1, 2, 3], [4, 5, 6, 7]])
    test2 = data.groupByFixedLength(2)
    assert test2.keys().collect() == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert allclose(test2.index, array([0, 1]))
    assert allclose(test2.values().collect(), [[0, 1], [2, 3], [4, 5], [6, 7]])


def test_mean_by_fixed_length():
    data = fromList([arange(8)])
    test1 = data.meanByFixedLength(4)
    assert test1.keys().collect() == [(0,)]
    assert allclose(test1.index, array([0, 1, 2, 3]))
    assert allclose(test1.values().collect(), [[2, 3, 4, 5]])
    test2 = data.meanByFixedLength(2)
    assert test2.keys().collect() == [(0,)]
    assert allclose(test2.index, array([0, 1]))
    assert allclose(test2.values().collect(), [[3, 4]])
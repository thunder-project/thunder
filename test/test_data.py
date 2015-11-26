import pytest
from numpy import allclose, array, arange

from thunder.data.series.readers import fromList

pytestmark = pytest.mark.usefixtures("context")


def test_get_missing():
    data = fromList([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    assert data.get(-1) is None


def test_get():
    data = fromList([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    assert allclose(data.get(1), [5, 6, 7, 8])


def test_get_errors():
    data = fromList([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    with pytest.raises(ValueError):
        data.get((1, 2))
    with pytest.raises(ValueError):
        data.getMany([0, (0, 0)])


def test_get_many():
    data = fromList([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    vals = data.getMany([0, -1, 1, 0])
    assert len(vals) == 4
    assert allclose(vals[0], array([1, 2, 3, 4]))
    assert allclose(vals[2], array([5, 6, 7, 8]))
    assert allclose(vals[3], array([1, 2, 3, 4]))
    assert vals[1] is None


def test_get_range():
    a0 = array([1, 2, 3, 4])
    a1 = array([5, 6, 7, 8])
    data = fromList([a0, a1])
    vals = data.getRange(slice(None))
    assert len(vals) == 2
    assert allclose(vals[0], a0)
    assert allclose(vals[1], a1)
    vals = data.getRange(slice(0, 1))
    assert allclose(vals[0], a0)
    vals = data.getRange(slice(1))
    assert allclose(vals[0], a0)
    vals = data.getRange(slice(1, 2))
    assert allclose(vals[0], a1)
    vals = data.getRange(slice(2, 3))
    assert len(vals) == 0


def test_get_range_error():
    data = fromList([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    with pytest.raises(ValueError):
        data.getRange([slice(1), slice(1)])
    with pytest.raises(ValueError):
        data.getRange(slice(1, 2, 2))


def test_brackets():
    a0 = array([1, 2, 3, 4])
    a1 = array([5, 6, 7, 8])
    data = fromList([a0, a1])
    assert allclose(data[0], a0)
    assert allclose(data[1], a1)
    assert allclose(data[0:1], [a0])
    assert allclose(data[:], [a0, a1])
    assert allclose(data[1:], [a1])
    assert allclose(data[:1], [a0])


def test_brackets_multi():
    a0 = array([1, 2])
    a1 = array([3, 4])
    a2 = array([5, 6])
    a3 = array([7, 8])
    data = fromList([a0, a1, a2, a3], keys=[(0, 0), (0, 1), (1, 0), (1, 1)])
    assert allclose(data[(0, 1)], a1)
    assert allclose(data[0, 1], a1)
    assert allclose(data[0:1, 1:2], a1)
    assert allclose(data[:4, :1], [a0, a2])
    assert allclose(data[:, 1:2], [a1, a3])
    assert allclose(data[:, :], [a0, a1, a2, a3])
    assert allclose(data[0, :], [a0, a1])


def test_dims():
    keys = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
            (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)]
    data = fromList(arange(12), keys=keys)
    dims = data.dims
    assert(allclose(dims.max, (2, 3, 2)))
    assert(allclose(dims.count, (2, 3, 2)))
    assert(allclose(dims.min, (1, 1, 1)))


def test_casting():
    data = fromList([array([1, 2, 3], 'int16')])
    assert data.astype('int64').collectValuesAsArray().dtype == 'int64'
    assert data.astype('float32').collectValuesAsArray().dtype == 'float32'
    assert data.astype('float64').collectValuesAsArray().dtype == 'float64'
    assert data.astype('float16', casting='unsafe').collectValuesAsArray().dtype == 'float16'


def test_casting_smallfloat():
    data = fromList([arange(3, dtype='uint8')])
    casted = data.astype("smallfloat")
    assert str(casted.dtype) == 'float16'
    assert str(casted.collectValuesAsArray().dtype) == 'float16'


def test_sort_by_key():
    data = fromList(arange(6), keys=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
    vals = data.sortByKey().keys().collect()
    assert allclose(vals, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)])
    data = fromList(arange(3), keys=[(0,), (2,), (1,)])
    vals = data.sortByKey().keys().collect()
    assert allclose(vals, [(0,), (1,), (2,)])
    data = fromList(arange(3), keys=[0, 2, 1])
    vals = data.sortByKey().keys().collect()
    assert allclose(vals, [0, 1, 2])


def test_collect():
    keys = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    data = fromList([[0], [1], [2], [3], [4], [5]], keys=keys)
    vals = data.collectKeysAsArray()
    assert allclose(vals, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    vals = data.collectValuesAsArray()
    assert allclose(vals, [[0], [1], [2], [3], [4], [5]])
    vals = data.collectValuesAsArray(sorting=True)
    assert allclose(vals, [[0], [3], [1], [4], [2], [5]])


def test_range_int_key():
    keys = [0, 1, 2, 3, 4, 5]
    data = fromList([[0], [1], [2], [3], [4], [5]], keys=keys)
    assert allclose(data.range(0, 2).collectKeysAsArray(), [0, 1])
    assert allclose(data.range(0, 5).collectKeysAsArray(), [0, 1, 2, 3, 4])


def test_range_tuple_key():
    keys = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    data = fromList([[0], [1], [2], [3], [4], [5]], keys=keys)
    vals = data.range((0, 0), (1, 1)).collectKeysAsArray()
    assert allclose(vals, [(0, 0), (0, 1), (0, 2), (1, 0)])


def test_mean():
    data = fromList([arange(8)])
    val = data.mean()
    expected = data.collectValuesAsArray().mean(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_sum():
    data = fromList([arange(8)])
    val = data.sum()
    expected = data.collectValuesAsArray().sum(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_variance():
    data = fromList([arange(8)])
    val = data.variance()
    expected = data.collectValuesAsArray().var(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_stdev():
    data = fromList([arange(8)])
    val = data.stdev()
    expected = data.collectValuesAsArray().std(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_stats():
    data = fromList([arange(8)])
    stats = data.stats()
    expected = data.collectValuesAsArray().mean(axis=0)
    assert allclose(stats.mean(), expected)
    expected = data.collectValuesAsArray().var(axis=0)
    assert allclose(stats.variance(), expected)


def test_max():
    data = fromList([arange(8)])
    val = data.max()
    expected = data.collectValuesAsArray().max(axis=0)
    assert allclose(val, expected)


def test_min():
    data = fromList([arange(8)])
    val = data.min()
    expected = data.collectValuesAsArray().min(axis=0)
    assert allclose(val, expected)

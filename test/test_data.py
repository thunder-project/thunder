import pytest
from numpy import allclose, array, arange

from thunder.data.series.readers import fromlist

pytestmark = pytest.mark.usefixtures("context")


def test_get_missing():
    data = fromlist([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    assert data.get(-1) is None


def test_get():
    data = fromlist([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    assert allclose(data.get(1), [5, 6, 7, 8])


def test_get_errors():
    data = fromlist([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    with pytest.raises(ValueError):
        data.get((1, 2))
    with pytest.raises(ValueError):
        data.getmany([0, (0, 0)])


def test_get_many():
    data = fromlist([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    vals = data.getmany([0, -1, 1, 0])
    assert len(vals) == 4
    assert allclose(vals[0], array([1, 2, 3, 4]))
    assert allclose(vals[2], array([5, 6, 7, 8]))
    assert allclose(vals[3], array([1, 2, 3, 4]))
    assert vals[1] is None


def test_get_range():
    a0 = array([1, 2, 3, 4])
    a1 = array([5, 6, 7, 8])
    data = fromlist([a0, a1])
    vals = data.getrange(slice(None))
    assert len(vals) == 2
    assert allclose(vals[0], a0)
    assert allclose(vals[1], a1)
    vals = data.getrange(slice(0, 1))
    assert allclose(vals[0], a0)
    vals = data.getrange(slice(1))
    assert allclose(vals[0], a0)
    vals = data.getrange(slice(1, 2))
    assert allclose(vals[0], a1)
    vals = data.getrange(slice(2, 3))
    assert len(vals) == 0


def test_get_range_error():
    data = fromlist([array([1, 2, 3, 4]), array([5, 6, 7, 8])])
    with pytest.raises(ValueError):
        data.getrange([slice(1), slice(1)])
    with pytest.raises(ValueError):
        data.getrange(slice(1, 2, 2))


def test_brackets():
    a0 = array([1, 2, 3, 4])
    a1 = array([5, 6, 7, 8])
    data = fromlist([a0, a1])
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
    data = fromlist([a0, a1, a2, a3], keys=[(0, 0), (0, 1), (1, 0), (1, 1)])
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
    data = fromlist(arange(12), keys=keys)
    dims = data.dims
    assert(allclose(dims.max, (2, 3, 2)))
    assert(allclose(dims.count, (2, 3, 2)))
    assert(allclose(dims.min, (1, 1, 1)))


def test_casting():
    data = fromlist([array([1, 2, 3], 'int16')])
    assert data.astype('int64').toarray().dtype == 'int64'
    assert data.astype('float32').toarray().dtype == 'float32'
    assert data.astype('float64').toarray().dtype == 'float64'
    assert data.astype('float16', casting='unsafe').toarray().dtype == 'float16'


def test_casting_smallfloat():
    data = fromlist([arange(3, dtype='uint8')])
    casted = data.astype("smallfloat")
    assert str(casted.dtype) == 'float16'
    assert str(casted.toarray().dtype) == 'float16'


def test_sort():
    data = fromlist(arange(6), keys=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
    vals = data.sort().keys().collect()
    assert allclose(vals, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)])
    data = fromlist(arange(3), keys=[(0,), (2,), (1,)])
    vals = data.sort().keys().collect()
    assert allclose(vals, [(0,), (1,), (2,)])
    data = fromlist(arange(3), keys=[0, 2, 1])
    vals = data.sort().keys().collect()
    assert allclose(vals, [0, 1, 2])


def test_collect():
    keys = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    data = fromlist([[0], [1], [2], [3], [4], [5]], keys=keys)
    assert allclose(data.keys().collect(), [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    assert allclose(data.toarray(), [0, 1, 2, 3, 4, 5])
    assert allclose(data.toarray(sorting=True), [0, 3, 1, 4, 2, 5])


def test_range_int_key():
    keys = [0, 1, 2, 3, 4, 5]
    data = fromlist([[0], [1], [2], [3], [4], [5]], keys=keys)
    assert allclose(data.range(0, 2).keys().collect(), [0, 1])
    assert allclose(data.range(0, 5).keys().collect(), [0, 1, 2, 3, 4])


def test_range_tuple_key():
    keys = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    data = fromlist([[0], [1], [2], [3], [4], [5]], keys=keys)
    vals = data.range((0, 0), (1, 1)).keys().collect()
    assert allclose(vals, [(0, 0), (0, 1), (0, 2), (1, 0)])


def test_mean():
    data = fromlist([arange(8), arange(8)])
    val = data.mean()
    expected = data.toarray().mean(axis=0)
    print(data.toarray())
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_sum():
    data = fromlist([arange(8), arange(8)])
    val = data.sum()
    expected = data.toarray().sum(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_var():
    data = fromlist([arange(8), arange(8)])
    val = data.var()
    expected = data.toarray().var(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_std():
    data = fromlist([arange(8), arange(8)])
    val = data.std()
    expected = data.toarray().std(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_stats():
    data = fromlist([arange(8), arange(8)])
    stats = data.stats()
    expected = data.toarray().mean(axis=0)
    assert allclose(stats.mean(), expected)
    expected = data.toarray().var(axis=0)
    assert allclose(stats.var(), expected)


def test_max():
    data = fromlist([arange(8), arange(8)])
    val = data.max()
    expected = data.toarray().max(axis=0)
    assert allclose(val, expected)


def test_min():
    data = fromlist([arange(8), arange(8)])
    val = data.min()
    expected = data.toarray().min(axis=0)
    assert allclose(val, expected)

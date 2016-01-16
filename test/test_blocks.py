import pytest
from numpy import arange, array, allclose, ones

from thunder.images.readers import fromlist

pytestmark = pytest.mark.usefixtures("engspark")


def test_conversion(engspark):
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a], engine=engspark)
    vals = data.toblocks((2, 2)).tordd().sortByKey().values().collect()
    truth = [array([a[0:2, 0:2], a[0:2, 0:2]]), array([a[2:4, 0:2], a[2:4, 0:2]])]
    assert allclose(vals, truth)


def test_full(engspark):
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a], engine=engspark)
    vals = data.toblocks((4, 2)).tordd().values().collect()
    truth = [a, a]
    assert allclose(vals, truth)


def test_count(engspark):
    a = arange(8).reshape((2, 4))
    data = fromlist([a], engine=engspark)
    assert data.toblocks((1, 1)).count() == 8
    assert data.toblocks((1, 2)).count() == 4
    assert data.toblocks((2, 2)).count() == 2
    assert data.toblocks((2, 4)).count() == 1


def test_conversion_series(engspark):
    a = arange(8).reshape((4, 2))
    data = fromlist([a], engine=engspark)
    vals = data.toblocks((1, 2)).toseries().toarray()
    assert allclose(vals, a)


def test_conversion_series_3d(engspark):
    a = arange(24).reshape((2, 3, 4))
    data = fromlist([a], engine=engspark)
    vals = data.toblocks((2, 3, 4)).toseries().toarray()
    assert allclose(vals, a)


def test_roundtrip(engspark):
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a], engine=engspark)
    vals = data.toblocks((2, 2)).toimages()
    assert allclose(vals.toarray(), data.toarray())


def test_series_roundtrip_simple(engspark):
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a], engine=engspark)
    vals = data.toseries().toimages()
    assert allclose(vals.toarray(), data.toarray())


def test_shape(engspark):
    data = fromlist([ones((30, 30)) for _ in range(0, 3)], engine=engspark)
    blocks = data.toblocks((10, 10))
    values = [v for k, v in blocks.tordd().collect()]
    assert blocks.subshape == (3, 10, 10)
    assert all([v.shape == (3, 10, 10) for v in values])
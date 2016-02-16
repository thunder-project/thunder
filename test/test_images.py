import pytest
from numpy import arange, allclose, array

from thunder.images.readers import fromlist
from thunder.images.images import Images
from thunder.series.series import Series
from thunder.series.timeseries import TimeSeries


pytestmark = pytest.mark.usefixtures("eng")


def test_map(eng):
    data = fromlist([arange(6).reshape((2, 3))], engine=eng)
    assert allclose(data.map(lambda x: x + 1).toarray(), [[1, 2, 3], [4, 5, 6]])


def test_map_singleton(eng):
    data = fromlist([arange(6).reshape((2, 3)), arange(6).reshape((2, 3))], engine=eng)
    mapped = data.map(lambda x: x.mean())
    assert mapped.shape == (2, 1)


def test_filter(eng):
    data = fromlist([arange(6).reshape((2, 3)), arange(6).reshape((2, 3)) * 2], engine=eng)
    assert allclose(data.filter(lambda x: x.sum() > 21).toarray(), [[0, 2, 4], [6, 8, 10]])


def test_sample(eng):
    data = fromlist([array([[1, 5], [1, 5]]), array([[1, 10], [1, 10]])], engine=eng)
    assert allclose(data.sample(2).shape, (2, 2, 2))
    assert allclose(data.sample(1).shape, (1, 2, 2))
    assert allclose(data.filter(lambda x: x.max() > 5).sample(1).toarray(), [[1, 10], [1, 10]])


def test_first(eng):
    data = fromlist([array([[1, 5], [1, 5]]), array([[1, 10], [1, 10]])], engine=eng)
    assert allclose(data.first(), [[1, 5], [1, 5]])


def test_squeeze(eng):
    data = fromlist([array([[1, 5], [1, 5]]), array([[1, 10], [1, 10]])], engine=eng)
    assert data.shape == (2, 2, 2)
    assert data[:, :, 0].shape == (2, 2, 1)
    assert data[:, 0, 0].shape == (2, 1, 1)
    assert data[:, :, 0].squeeze().shape == (2, 2)
    assert data[:, 0, 0].squeeze().shape == (2, 1)


def test_toseries(eng):
    data = fromlist([arange(6).reshape((2, 3))], engine=eng)
    truth = [[0, 1, 2], [3, 4, 5]]
    assert isinstance(data.toseries(), Series)
    assert allclose(data.toseries().toarray(), truth)


def test_toseries_roundtrip(eng):
    data = fromlist([arange(6).reshape((2, 3)), arange(6).reshape((2, 3))], engine=eng)
    assert isinstance(data.toseries(), Series)
    assert isinstance(data.toseries().toimages(), Images)
    assert allclose(data.toseries().toimages().toarray(), data.toarray())


def test_totimeseries(eng):
    data = fromlist([arange(6).reshape((2, 3))], engine=eng)
    assert isinstance(data.totimeseries(), TimeSeries)
    assert allclose(data.toseries().toarray(), data.totimeseries().toarray())


def test_toseries_pack_2d(eng):
    original = arange(6).reshape((2, 3))
    data = fromlist([original], engine=eng)
    assert allclose(data.toseries().toarray(), original)


def test_toseries_pack_3d(eng):
    original = arange(24).reshape((2, 3, 4))
    data = fromlist([original], engine=eng)
    assert allclose(data.toseries().toarray(), original)


def test_subsample(eng):
    data = fromlist([arange(24).reshape((4, 6))], engine=eng)
    vals = data.subsample(2).toarray()
    truth = [[0, 2, 4], [12, 14, 16]]
    assert allclose(vals, truth)


def test_median_filter_2d(eng):
    data = fromlist([arange(24).reshape((4, 6))], engine=eng)
    assert data.median_filter(2).toarray().shape == (4, 6)
    assert data.median_filter([2, 2]).toarray().shape == (4, 6)
    assert allclose(data.median_filter(2).toarray(), data.median_filter([2, 2]).toarray())


def test_median_filter_3d(eng):
    data = fromlist([arange(24).reshape((2, 3, 4))], engine=eng)
    assert data.median_filter(2).toarray().shape == (2, 3, 4)
    assert data.median_filter([2, 2, 2]).toarray().shape == (2, 3, 4)
    assert data.median_filter([2, 2, 0]).toarray().shape == (2, 3, 4)
    assert allclose(data.median_filter(2).toarray(), data.median_filter([2, 2, 2]).toarray())


def test_median_filter_3d_empty(eng):
    data = fromlist([arange(24).reshape((2, 3, 4))], engine=eng)
    test1 = data.median_filter([2, 2, 0])[:, :, :, 0].squeeze().toarray()
    test2 = data[:, :, :, 0].squeeze().median_filter([2, 2]).toarray()
    assert test1.shape == (2, 3)
    assert test2.shape == (2, 3)
    assert allclose(test1, test2)


def test_gaussian_filter_2d(eng):
    data = fromlist([arange(24).reshape((4, 6))], engine=eng)
    assert data.gaussian_filter(2).toarray().shape == (4, 6)
    assert data.gaussian_filter([2, 2]).toarray().shape == (4, 6)
    assert allclose(data.gaussian_filter(2).toarray(), data.gaussian_filter([2, 2]).toarray())


def test_gaussian_filter_3d(eng):
    data = fromlist([arange(24).reshape((2, 3, 4))], engine=eng)
    assert data.gaussian_filter(2).toarray().shape == (2, 3, 4)
    assert data.gaussian_filter([2, 2, 2]).toarray().shape == (2, 3, 4)
    assert data.gaussian_filter([2, 2, 0]).toarray().shape == (2, 3, 4)
    assert allclose(data.gaussian_filter(2).toarray(), data.gaussian_filter([2, 2, 2]).toarray())


def test_uniform_filter_2d(eng):
    data = fromlist([arange(24).reshape((4, 6))], engine=eng)
    assert data.uniform_filter(2).toarray().shape == (4, 6)
    assert data.uniform_filter([2, 2]).toarray().shape == (4, 6)
    assert allclose(data.uniform_filter(2).toarray(), data.uniform_filter([2, 2]).toarray())


def test_uniform_filter_3d(eng):
    data = fromlist([arange(24).reshape((2, 3, 4))], engine=eng)
    assert data.uniform_filter(2).toarray().shape == (2, 3, 4)
    assert data.uniform_filter([2, 2, 2]).toarray().shape == (2, 3, 4)
    assert data.uniform_filter([2, 2, 0]).toarray().shape == (2, 3, 4)
    assert allclose(data.uniform_filter(2).toarray(), data.uniform_filter([2, 2, 2]).toarray())


def test_mean(eng):
    original = arange(24).reshape((2, 3, 4))
    data = fromlist(list(original), engine=eng)
    assert allclose(data.mean().shape, (1, 3, 4))
    assert allclose(data.mean().toarray(), original.mean(axis=0))


def test_sum(eng):
    original = arange(24).reshape((2, 3, 4))
    data = fromlist(list(original), engine=eng)
    assert allclose(data.sum().shape, (1, 3, 4))
    assert allclose(data.sum().toarray(), original.sum(axis=0))


def test_var(eng):
    original = arange(24).reshape((2, 3, 4))
    data = fromlist(list(original), engine=eng)
    assert allclose(data.var().shape, (1, 3, 4))
    assert allclose(data.var().toarray(), original.var(axis=0))


def test_subtract(eng):
    original = arange(24).reshape((4, 6))
    data = fromlist([original], engine=eng)
    assert allclose(data.subtract(1).toarray(), original - 1)
    sub = arange(24).reshape((4, 6))
    assert allclose(data.subtract(sub).toarray(), original - sub)


# def test_localcorr(eng):
#     imgs = [
#         array([[1.0, 2.0, 9.0], [5.0, 4.0, 5.0], [4.0, 6.0, 0.0]]),
#         array([[2.0, 2.0, 2.0], [2.0, 2.0, 4.0], [2.0, 3.0, 2.0]]),
#         array([[3.0, 4.0, 1.0], [5.0, 8.0, 1.0], [6.0, 2.0, 1.0]])
#     ]
#     data = fromlist(imgs, engine=eng)
#     vals = data.localcorr(1)
#     truth = corrcoef(map(lambda i: i.mean(), imgs), array([4.0, 2.0, 8.0]))[0, 1]
#     assert allclose(vals[1][1], truth)
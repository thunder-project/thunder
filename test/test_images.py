import pytest
from numpy import arange, allclose, array, corrcoef

from thunder.data.images.readers import fromlist

pytestmark = pytest.mark.usefixtures("eng")


def test_toseries(eng):
    data = fromlist([arange(6).reshape((2, 3))], engine=eng)
    truth = [0, 3, 1, 4, 2, 5]
    vals = data.toseries().toarray()
    assert allclose(vals, truth)
    #vals = data.toblocks((1, 1)).toseries().toarray()
    #assert allclose(vals, truth)

#
# def test_totimeseries(eng):
#     data = fromlist([arange(6).reshape((2, 3))], engine=eng)
#     vals1 = data.totimeseries().toarray()
#     vals2 = data.totimeseries().toarray()
#     assert allclose(vals1, vals2)
#
#
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


def test_gaussian_filter_2d(eng):
    data = fromlist([arange(24).reshape((4, 6))], engine=eng)
    assert data.gaussian_filter(2).toarray().shape == (4, 6)
    assert data.gaussian_filter([2, 2]).toarray().shape == (4, 6)
    assert allclose(data.gaussian_filter(2).toarray(), data.gaussian_filter([2, 2]).toarray())


def test_gaussian_filter_3d(eng):
    data = fromlist([arange(24).reshape((2, 3, 4))], engine=eng)
    assert data.gaussian_filter(2).toarray().shape == (2, 3, 4)
    assert data.gaussian_filter([2, 2, 2]).toarray().shape == (2, 3, 4)


def test_uniform_filter_2d(eng):
    data = fromlist([arange(24).reshape((4, 6))], engine=eng)
    assert data.uniform_filter(2).toarray().shape == (4, 6)
    assert data.uniform_filter([2, 2]).toarray().shape == (4, 6)
    assert allclose(data.uniform_filter(2).toarray(), data.uniform_filter([2, 2]).toarray())


def test_uniform_filter_3d(eng):
    data = fromlist([arange(24).reshape((2, 3, 4))], engine=eng)
    assert data.uniform_filter(2).toarray().shape == (2, 3, 4)
    assert data.uniform_filter([2, 2, 2]).toarray().shape == (2, 3, 4)


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


def test_crop(eng):
    original = arange(24).reshape((4, 6))
    data = fromlist([original], engine=eng)
    assert allclose(data.crop((0, 0), (2, 2)).toarray(), original[0:2, 0:2])
    assert allclose(data.crop((0, 1), (2, 2)).toarray(), original[0:2, 1:2])


def test_planes(eng):
    original = arange(24).reshape((2, 3, 4))
    data = fromlist([original], engine=eng)
    assert allclose(data.planes(0, 1).toarray(), original[:, :, 0:1])


def test_subtract(eng):
    original = arange(24).reshape((4, 6))
    data = fromlist([original], engine=eng)
    assert allclose(data.subtract(1).toarray(), original - 1)
    sub = arange(24).reshape((4, 6))
    assert allclose(data.subtract(sub).toarray(), original - sub)

#
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
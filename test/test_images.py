import pytest
from numpy import arange, allclose, array, corrcoef

from thunder.data.images.readers import fromList

pytestmark = pytest.mark.usefixtures("context")


def test_toseries():
    data = fromList([arange(6).reshape((2, 3))])
    truth = [0, 3, 1, 4, 2, 5]
    vals = data.toSeries().toarray()
    assert allclose(vals, truth)
    vals = data.toBlocks((1, 1)).toSeries().toarray()
    assert allclose(vals, truth)


def test_totimeseries():
    data = fromList([arange(6).reshape((2, 3))])
    vals1 = data.toTimeSeries().toarray()
    vals2 = data.toTimeSeries().toarray()
    assert allclose(vals1, vals2)


def test_toseries_pack_2d():
    original = arange(6).reshape((2, 3))
    data = fromList([original])
    assert allclose(data.toSeries().pack(), original)


def test_toseries_pack_3d():
    original = arange(24).reshape((2, 3, 4))
    data = fromList([original])
    assert allclose(data.toSeries().pack(), original)


def test_subsample():
    data = fromList([arange(24).reshape((4, 6))])
    vals = data.subsample(2).toarray()
    truth = [[0, 2, 4], [12, 14, 16]]
    assert allclose(vals, truth)


def test_median_filter_2d():
    data = fromList([arange(24).reshape((4, 6))])
    assert data.medianFilter(2).toarray().shape == (4, 6)
    assert data.medianFilter([2, 2]).toarray().shape == (4, 6)
    assert allclose(data.medianFilter(2).toarray(), data.medianFilter([2, 2]).toarray())


def test_median_filter_3d():
    data = fromList([arange(24).reshape((2, 3, 4))])
    assert data.medianFilter(2).toarray().shape == (2, 3, 4)
    assert data.medianFilter([2, 2, 2]).toarray().shape == (2, 3, 4)


def test_gaussian_filter_2d():
    data = fromList([arange(24).reshape((4, 6))])
    assert data.gaussianFilter(2).toarray().shape == (4, 6)
    assert data.gaussianFilter([2, 2]).toarray().shape == (4, 6)
    assert allclose(data.gaussianFilter(2).toarray(), data.gaussianFilter([2, 2]).toarray())


def test_gaussian_filter_3d():
    data = fromList([arange(24).reshape((2, 3, 4))])
    assert data.gaussianFilter(2).toarray().shape == (2, 3, 4)
    assert data.gaussianFilter([2, 2, 2]).toarray().shape == (2, 3, 4)


def test_uniform_filter_2d():
    data = fromList([arange(24).reshape((4, 6))])
    assert data.uniformFilter(2).toarray().shape == (4, 6)
    assert data.uniformFilter([2, 2]).toarray().shape == (4, 6)
    assert allclose(data.uniformFilter(2).toarray(), data.uniformFilter([2, 2]).toarray())


def test_uniform_filter_3d():
    data = fromList([arange(24).reshape((2, 3, 4))])
    assert data.uniformFilter(2).toarray().shape == (2, 3, 4)
    assert data.uniformFilter([2, 2, 2]).toarray().shape == (2, 3, 4)


def test_mean():
    original = arange(24).reshape((2, 3, 4))
    data = fromList(list(original))
    assert allclose(data.mean(), original.mean(axis=0))


def test_sum():
    original = arange(24).reshape((2, 3, 4))
    data = fromList(list(original))
    assert allclose(data.sum(), original.sum(axis=0))


def test_var():
    original = arange(24).reshape((2, 3, 4))
    data = fromList(list(original))
    assert allclose(data.var(), original.var(axis=0))


def test_crop():
    original = arange(24).reshape((4, 6))
    data = fromList([original])
    assert allclose(data.crop((0, 0), (2, 2)).values().first(), original[0:2, 0:2])
    assert allclose(data.crop((0, 1), (2, 2)).values().first(), original[0:2, 1:2])


def test_planes():
    original = arange(24).reshape((2, 3, 4))
    data = fromList([original])
    assert allclose(data.planes(0, 1).values().first(), original[:, :, 0:1])


def test_subtract():
    original = arange(24).reshape((4, 6))
    data = fromList([original])
    assert allclose(data.subtract(1).values().first(), original - 1)
    sub = arange(24).reshape((4, 6))
    assert allclose(data.subtract(sub).values().first(), original - sub)


def test_localcorr():
    imgs = [
        array([[1.0, 2.0, 9.0], [5.0, 4.0, 5.0], [4.0, 6.0, 0.0]]),
        array([[2.0, 2.0, 2.0], [2.0, 2.0, 4.0], [2.0, 3.0, 2.0]]),
        array([[3.0, 4.0, 1.0], [5.0, 8.0, 1.0], [6.0, 2.0, 1.0]])
    ]
    data = fromList(imgs)
    vals = data.localcorr(1)
    truth = corrcoef(map(lambda i: i.mean(), imgs), array([4.0, 2.0, 8.0]))[0, 1]
    assert allclose(vals[1][1], truth)
import pytest
from numpy import arange, array, allclose, ones

from thunder.data.images.readers import fromList

pytestmark = pytest.mark.usefixtures("context")


def test_blocks_pixels():
    a = arange(8).reshape((4, 2))
    data = fromList([a, a])
    vals = data.toBlocks((2, 2), units='pixels').values().collect()
    truth = [array([a[0:2, 0:2], a[0:2, 0:2]]), array([a[2:4, 0:2], a[2:4, 0:2]])]
    assert allclose(vals, truth)


def test_blocks_splits():
    a = arange(8).reshape((4, 2))
    data = fromList([a, a])
    vals = data.toBlocks((2, 1), units='splits').values().collect()
    truth = [array([a[0:2, 0:2], a[0:2, 0:2]]), array([a[2:4, 0:2], a[2:4, 0:2]])]
    assert allclose(vals, truth)


def test_blocks_pixels_full():
    a = arange(8).reshape((4, 2))
    data = fromList([a, a])
    vals = data.toBlocks((4, 2), units='pixels').values().collect()
    truth = [a, a]
    assert allclose(vals, truth)


def test_blocks_splits_full():
    a = arange(8).reshape((4, 2))
    data = fromList([a, a])
    vals = data.toBlocks((1, 1), units='splits').values().collect()
    truth = [a, a]
    assert allclose(vals, truth)


def test_blocks_splits_count():
    a = arange(8).reshape((2, 4))
    data = fromList([a])
    assert data.toBlocks((2, 4), units='splits').count() == 8


def test_blocks_pixels_count():
    a = arange(8).reshape((2, 4))
    data = fromList([a])
    assert data.toBlocks((1, 1), units='pixels').count() == 8


def test_blocks_conversion():
    a = arange(8).reshape((4, 2))
    data = fromList([a])
    vals = data.toBlocks((1, 2), units='splits').toSeries().pack()
    assert allclose(vals, a)


def test_blocks_conversion_3d():
    a = arange(24).reshape((2, 3, 4))
    data = fromList([a])
    vals = data.toBlocks((2, 3, 4), units='splits').toSeries().pack()
    assert allclose(vals, a)


def test_padded_blocks_conversion():
    a = arange(8).reshape((4, 2))
    data = fromList([a])
    vals = data.toBlocks((1, 2), padding=(1, 1), units='splits').toSeries().pack()
    assert allclose(vals, a)


def test_blocks_roundtrip():
    a = arange(8).reshape((4, 2))
    data = fromList([a, a])
    vals = data.toBlocks((2, 2)).toImages()
    assert allclose(vals.collectValuesAsArray(), data.collectValuesAsArray())


def test_padded_blocks_roundtrip():
    a = arange(8).reshape((4, 2))
    data = fromList([a, a])
    vals = data.toBlocks((2, 2), padding=(2, 2)).toImages()
    assert allclose(vals.collectValuesAsArray(), data.collectValuesAsArray())


def test_blocks_shape():
    data = fromList([ones((30, 30)) for _ in range(0, 3)])
    blocks = data.toBlocks((10, 10)).collect()
    keys = [k for k, v in blocks]
    assert all(k.pixelsPerDim == (10, 10) for k in keys)
    assert all(k.spatialShape == (10, 10) for k in keys)


def test_blocks_neighbors():
    data = fromList([ones((30, 30)) for _ in range(0, 3)])
    blocks = data.toBlocks((10, 10)).collect()
    keys = [k for k, v in blocks]
    assert keys[0].neighbors() == [(0, 10), (10, 0), (10, 10)]
    assert keys[1].neighbors() == [(0, 0), (0, 10), (10, 10), (20, 0), (20, 10)]
    assert keys[2].neighbors() == [(10, 0), (10, 10), (20, 10)]
    assert keys[3].neighbors() == [(0, 0), (0, 20), (10, 0), (10, 10), (10, 20)]
    assert keys[4].neighbors() == [(0, 0), (0, 10), (0, 20), (10, 0), (10, 20), (20, 0), (20, 10), (20, 20)]
    assert keys[5].neighbors() == [(10, 0), (10, 10), (10, 20), (20, 0), (20, 20)]
    assert keys[6].neighbors() == [(0, 10), (10, 10), (10, 20)]
    assert keys[7].neighbors() == [(0, 10), (0, 20), (10, 10), (20, 10), (20, 20)]
    assert keys[8].neighbors() == [(10, 10), (10, 20), (20, 10)]
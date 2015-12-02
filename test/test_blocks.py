import pytest
from numpy import arange, array, allclose, ones

from thunder.data.images.readers import fromlist

pytestmark = pytest.mark.usefixtures("context")


def test_blocks_pixels():
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a])
    vals = data.toblocks((2, 2), units='pixels').values().collect()
    truth = [array([a[0:2, 0:2], a[0:2, 0:2]]), array([a[2:4, 0:2], a[2:4, 0:2]])]
    assert allclose(vals, truth)


def test_blocks_splits():
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a])
    vals = data.toblocks((2, 1), units='splits').values().collect()
    truth = [array([a[0:2, 0:2], a[0:2, 0:2]]), array([a[2:4, 0:2], a[2:4, 0:2]])]
    assert allclose(vals, truth)


def test_blocks_pixels_full():
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a])
    vals = data.toblocks((4, 2), units='pixels').values().collect()
    truth = [a, a]
    assert allclose(vals, truth)


def test_blocks_splits_full():
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a])
    vals = data.toblocks((1, 1), units='splits').values().collect()
    truth = [a, a]
    assert allclose(vals, truth)


def test_blocks_splits_count():
    a = arange(8).reshape((2, 4))
    data = fromlist([a])
    assert data.toblocks((2, 4), units='splits').count() == 8


def test_blocks_pixels_count():
    a = arange(8).reshape((2, 4))
    data = fromlist([a])
    assert data.toblocks((1, 1), units='pixels').count() == 8


def test_blocks_conversion():
    a = arange(8).reshape((4, 2))
    data = fromlist([a])
    vals = data.toblocks((1, 2), units='splits').toseries().pack()
    assert allclose(vals, a)


def test_blocks_conversion_3d():
    a = arange(24).reshape((2, 3, 4))
    data = fromlist([a])
    vals = data.toblocks((2, 3, 4), units='splits').toseries().pack()
    assert allclose(vals, a)


def test_padded_blocks_conversion():
    a = arange(8).reshape((4, 2))
    data = fromlist([a])
    vals = data.toblocks((1, 2), padding=(1, 1), units='splits').toseries().pack()
    assert allclose(vals, a)


def test_blocks_roundtrip():
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a])
    vals = data.toblocks((2, 2)).toimages()
    assert allclose(vals.toarray(), data.toarray())


def test_padded_blocks_roundtrip():
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a])
    vals = data.toblocks((2, 2), padding=(2, 2)).toimages()
    assert allclose(vals.toarray(), data.toarray())


def test_blocks_shape():
    data = fromlist([ones((30, 30)) for _ in range(0, 3)])
    blocks = data.toblocks((10, 10)).collect()
    keys = [k for k, v in blocks]
    assert all(k.pixels_per_dim == (10, 10) for k in keys)
    assert all(k.block_shape == (10, 10) for k in keys)


def test_blocks_neighbors():
    data = fromlist([ones((30, 30)) for _ in range(0, 3)])
    blocks = data.toblocks((10, 10)).collect()
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
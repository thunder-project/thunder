import pytest
from numpy import arange, array, allclose

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
    original = arange(8).reshape((2, 4))
    data = fromList([original])
    assert data.toBlocks((2, 4), units='splits').count() == 8


def test_blocks_pixels_count():
    original = arange(8).reshape((2, 4))
    data = fromList([original])
    assert data.toBlocks((1, 1), units='pixels').count() == 8


def test_blocks_conversion():
    original = arange(8).reshape((4, 2))
    data = fromList([original])
    vals = data.toBlocks((1, 2), units='splits').toSeries().pack()
    assert allclose(vals, original)


def test_blocks_conversion_3d():
    original = arange(24).reshape((2, 3, 4))
    data = fromList([original])
    vals = data.toBlocks((2, 3, 4), units='splits').toSeries().pack()
    assert allclose(vals, original)


def test_padded_blocks_conversion():
    original = arange(8).reshape((4, 2))
    data = fromList([original])
    vals = data.toBlocks((1, 2), padding=(1, 1), units='splits').toSeries().pack()
    assert allclose(vals, original)


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
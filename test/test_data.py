import pytest
from numpy import allclose, array

from thunder import series, images

pytestmark = pytest.mark.usefixtures("eng")


def test_first(eng):
    data = series.fromlist([array([1, 2, 3]), array([4, 5, 6])], engine=eng)
    assert allclose(data.first(), [1, 2, 3])
    data = images.fromlist([array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]])], engine=eng)
    assert allclose(data.first(), [[1, 2], [3, 4]])


def test_casting(eng):
    data = series.fromlist([array([1, 2, 3], 'int16')], engine=eng)
    assert data.astype('int64').toarray().dtype == 'int64'
    assert data.astype('float32').toarray().dtype == 'float32'
    assert data.astype('float64').toarray().dtype == 'float64'
    assert data.astype('float16', casting='unsafe').toarray().dtype == 'float16'


def test_toarray(eng):
    original = [array([1, 2, 3]), array([4, 5, 6])]
    data = series.fromlist(original, engine=eng)
    assert allclose(data.toarray(), original)
    original = [array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]])]
    data = images.fromlist(original, engine=eng)
    assert allclose(data.toarray(), original)
import pytest
from numpy import allclose, array, asarray, add

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


def test_elementwise(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat2raw = asarray([[7, 8, 9], [10, 11, 12]])
    mat1 = series.fromlist(mat1raw, engine=eng)
    mat2 = series.fromlist(mat2raw, engine=eng)
    result = mat1.element_wise(mat2, add)
    truth = mat1raw + mat2raw
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, range(3))


def test_elementwise_scalar(eng):
    matraw = asarray([[1, 2, 3], [4, 5, 6]])
    mat = series.fromlist(matraw, engine=eng)
    result = mat.element_wise(2, add)
    truth = matraw + 2
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, range(3))


def test_elementwise_plus(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat2raw = asarray([[7, 8, 9], [10, 11, 12]])
    mat1 = series.fromlist(mat1raw, engine=eng)
    mat2 = series.fromlist(mat2raw, engine=eng)
    result = mat1.plus(mat2)
    truth = mat1raw + mat2raw
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, range(3))
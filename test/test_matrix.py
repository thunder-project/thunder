import pytest
from numpy import array, allclose, add
from thunder.data.series.readers import fromList

pytestmark = pytest.mark.usefixtures("context")


def test_elementwise():
    mat1 = fromList([array([1, 2, 3]), array([4, 5, 6])]).toMatrix()
    mat2 = fromList([array([7, 8, 9]), array([10, 11, 12])]).toMatrix()
    result = mat1.elementwise(mat2, add).collectValuesAsArray()
    truth = array([[8, 10, 12], [14, 16, 18]])
    assert allclose(result, truth)


def test_elementwise_array():
    mat = fromList([array([1, 2, 3])]).toMatrix()
    assert allclose(mat.elementwise(2, add).collectValuesAsArray(), array([3, 4, 5]))


def test_times_rdd():
    mat1 = fromList([array([1, 2, 3]), array([4, 5, 6])]).toMatrix()
    mat2 = fromList([array([7, 8, 9]), array([10, 11, 12])]).toMatrix()
    truth = array([[47, 52, 57], [64, 71, 78], [81, 90, 99]])
    resultA = mat1.times(mat2)
    assert allclose(resultA, truth)


def test_times_array():
    mat1 = fromList([array([1, 2, 3]), array([4, 5, 6])]).toMatrix()
    mat2 = array([[7, 8], [9, 10], [11, 12]])
    truth = [array([58, 64]), array([139, 154])]
    rdd = mat1.times(mat2)
    result = rdd.collectValuesAsArray()
    assert allclose(result, truth)
    assert allclose(rdd.index, range(0, 2))


def test_outer():
    mat1 = fromList([array([1, 2, 3]), array([4, 5, 6])]).toMatrix()
    resultA = mat1.gramian()
    resultB1 = mat1.gramian("accum")
    resultB2 = mat1.gramian("aggregate")
    truth = array([[17, 22, 27], [22, 29, 36], [27, 36, 45]])
    assert allclose(resultA, truth)
    assert allclose(resultB1, truth)
    assert allclose(resultB2, truth)

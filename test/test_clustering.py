import pytest
from numpy import array, allclose

from thunder.clustering.kmeans import KMeans, KMeansModel
from thunder.data import series

pytestmark = pytest.mark.usefixtures("context")


def test_kmeans_k1():

    dataLocal = [
        array([1.0, 2.0, 6.0]),
        array([1.0, 3.0, 0.0]),
        array([1.0, 4.0, 6.0])
    ]

    data = series.fromList(dataLocal)
    model = KMeans(k=1, maxIterations=20).fit(data)
    labels = model.predict(data)
    assert allclose(model.centers[0], array([1.0, 3.0, 4.0]))
    assert allclose(labels.values().collect(), array([0, 0, 0]))


def test_kmeans_k2():

    data, centers = KMeans.make(shape=(50, 5), k=2, npartitions=5, seed=42, withparams=True)
    truth = KMeansModel(centers)

    model = KMeans(k=2, maxIterations=20).fit(data)

    labels = array(model.predict(data).values().collect())
    labelsTrue = array(truth.predict(data).values().collect())
    assert allclose(labels, labelsTrue) or allclose(labels, 1 - labelsTrue)

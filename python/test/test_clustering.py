import shutil
import tempfile
from numpy import array, array_equal
from thunder.clustering.kmeans import KMeans, KMeansModel
from thunder.utils.datasets import DataSets
from thunder.rdds.series import Series
from test_utils import PySparkTestCase


class ClusteringTestCase(PySparkTestCase):
    def setUp(self):
        super(ClusteringTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(ClusteringTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestKMeans(ClusteringTestCase):
    def test_kmeans_k1(self):
        """ With k=1 always get one cluster centered on the mean"""

        data_local = [
            array([1.0, 2.0, 6.0]),
            array([1.0, 3.0, 0.0]),
            array([1.0, 4.0, 6.0])]

        data = Series(self.sc.parallelize(zip(range(1, 4), data_local)))

        model = KMeans(k=1, maxIterations=20).fit(data)
        labels = model.predict(data)
        assert array_equal(model.centers[0], array([1.0, 3.0, 4.0]))
        assert array_equal(labels.values().collect(), array([0, 0, 0]))

    def test_kmeans_k2(self):
        """ Test k=2 also with more points"""

        data, centerstrue = DataSets.make(self.sc, "kmeans",
                                          k=2, nrecords=50, npartitions=5, seed=42, returnParams=True)
        centerstrue = KMeansModel(centerstrue)

        model = KMeans(k=2, maxIterations=20).fit(data)

        labels = array(model.predict(data).values().collect())
        labelstrue = array(centerstrue.predict(data).values().collect())

        assert(array_equal(labels, labelstrue) or array_equal(labels, 1 - labelstrue))




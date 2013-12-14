import os
import shutil
import tempfile
from thunder.clustering.kmeans import kmeans
from thunder.util.dataio import parse
from test_utils import PySparkTestCase

# Hack to find the data files:
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
IRIS_DATA = os.path.join(DATA_DIR, "iris.txt")


def get_data_kmeans(self):
    return parse(self.sc.textFile(IRIS_DATA), "raw")


# For now, this only tests that the jobs run without crashing:
class ClusteringTestCase(PySparkTestCase):
    def setUp(self):
        super(ClusteringTestCase, self).setUp()
        self.outputDir = tempfile.mkdtemp()

    def tearDown(self):
        super(ClusteringTestCase, self).tearDown()
        shutil.rmtree(self.outputDir)


class TestKMeans(ClusteringTestCase):

    def test_kmeans(self):
        data = get_data_kmeans(self)
        labels, centers, dists, normDists = kmeans(data, 5, "euclidean")
        labels.collect()
        dists.collect()
        normDists.collect()


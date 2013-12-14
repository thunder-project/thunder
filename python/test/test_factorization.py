import os
import shutil
import tempfile
from thunder.factorization.pca import pca
from thunder.factorization.ica import ica
from thunder.factorization.rpca import rpca
from thunder.util.dataio import parse
from test_utils import PySparkTestCase

# Hack to find the data files:
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
IRIS_DATA = os.path.join(DATA_DIR, "iris.txt")
ICA_DATA = os.path.join(DATA_DIR, "ica.txt")
RPCA_DATA = os.path.join(DATA_DIR, "rpca.txt")


def get_data_pca(self):
    return parse(self.sc.textFile(IRIS_DATA), "raw")


def get_data_ica(self):
    return parse(self.sc.textFile(ICA_DATA), "raw")


def get_data_rpca(self):
    return parse(self.sc.textFile(RPCA_DATA), "raw")


# For now, this only tests that the jobs run without crashing:
class FactorizationTestCase(PySparkTestCase):
    def setUp(self):
        super(FactorizationTestCase, self).setUp()
        self.outputDir = tempfile.mkdtemp()

    def tearDown(self):
        super(FactorizationTestCase, self).tearDown()
        shutil.rmtree(self.outputDir)


class TestPCA(FactorizationTestCase):

    def test_pca(self):
        data = get_data_pca(self)
        pca(data, 4)


class TestICA(FactorizationTestCase):

    def test_ica(self):
        data = get_data_ica(self)
        ica(data, 4, 4)


class TestRPCA(FactorizationTestCase):

    def test_rpca(self):
        data = get_data_rpca(self)
        rpca(data)


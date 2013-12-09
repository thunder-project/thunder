import os
import shutil
import tempfile
from thunder.summary.fourier import fourier
from thunder.summary.localcorr import localcorr
from thunder.summary.query import query
from thunder.summary.ref import ref
from test_utils import PySparkTestCase

# Hack to find the data files:
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
FISH_DATA = os.path.join(DATA_DIR, "fish.txt")
QUERY_INDS = os.path.join(DATA_DIR, "summary/fish_inds.mat")


# For now, this only tests that the jobs run without crashing:
class SummaryTestCase(PySparkTestCase):
    def setUp(self):
        super(SummaryTestCase, self).setUp()
        self.outputDir = tempfile.mkdtemp()

    def tearDown(self):
        super(SummaryTestCase, self).tearDown()
        shutil.rmtree(self.outputDir)


class TestFourier(SummaryTestCase):

    def test_fourier(self):
        fourier(self.sc, FISH_DATA, self.outputDir, 1)


class TestLocalcorr(SummaryTestCase):

    def test_localcorr(self):
        localcorr(self.sc, FISH_DATA, self.outputDir, 5, 88, 76)


class TestQuery(SummaryTestCase):

    def test_query(self):
        query(self.sc, FISH_DATA, self.outputDir, QUERY_INDS, 88, 76)


class TestRef(SummaryTestCase):

    def test_ref_mean(self):
        ref(self.sc, FISH_DATA, self.outputDir, "mean")

    def test_ref_median(self):
        ref(self.sc, FISH_DATA, self.outputDir, "median")

    def test_ref_std(self):
        ref(self.sc, FISH_DATA, self.outputDir, "std")

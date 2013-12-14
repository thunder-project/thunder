import os
import shutil
import tempfile
from thunder.summary.fourier import fourier
from thunder.summary.localcorr import localcorr
from thunder.summary.query import query
from thunder.summary.ref import ref
from thunder.util.dataio import parse
from test_utils import PySparkTestCase

# Hack to find the data files:
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
FISH_DATA = os.path.join(DATA_DIR, "fish.txt")
QUERY_INDS = os.path.join(DATA_DIR, "summary/fish_inds.mat")


def get_data_fourier(self):
    return parse(self.sc.textFile(FISH_DATA), "dff").cache()


def get_data_localcorr(self):
    return parse(self.sc.textFile(FISH_DATA), "raw", "xyz").cache()


def get_data_query(self):
    return parse(self.sc.textFile(FISH_DATA), "dff", "linear", None, [88, 76]).cache()


def get_data_ref(self):
    return parse(self.sc.textFile(FISH_DATA), "raw", "xyz").cache()


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
        data = get_data_fourier(self)
        co, ph = fourier(data, 1)
        co.collect()
        ph.collect()


class TestLocalCorr(SummaryTestCase):

    def test_localcorr(self):
        data = get_data_localcorr(self)
        corrs, x, y = localcorr(data, 5, 88, 76)
        corrs.collect()
        x.collect()
        y.collect()


class TestQuery(SummaryTestCase):

    def test_query(self):
        data = get_data_query(self)
        ts = query(data, QUERY_INDS)


class TestRef(SummaryTestCase):

    def test_ref_mean(self):
        data = get_data_ref(self)
        refout, zinds = ref(data, "mean")
        refout.collect()


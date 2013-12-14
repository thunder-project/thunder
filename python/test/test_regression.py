import os
import shutil
import tempfile
from thunder.regression.regress import regress
from thunder.regression.shotgun import shotgun
from thunder.regression.tuning import tuning
from thunder.util.dataio import parse
from test_utils import PySparkTestCase

# Hack to find the data files:
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
FISH_DATA = os.path.join(DATA_DIR, "fish.txt")
SHOTGUN_DATA = os.path.join(DATA_DIR, "shotgun.txt")
SHOTGUN_MODEL = os.path.join(DATA_DIR, "regression/shotgun")
FISH_LINEAR_MODEL = os.path.join(DATA_DIR, "regression/fish_linear")
FISH_BILINEAR_MODEL = os.path.join(DATA_DIR, "regression/fish_bilinear")


def get_data_regression(self):
    return parse(self.sc.textFile(FISH_DATA), "dff").cache()


def get_data_shotgun(self):
    return parse(self.sc.textFile(SHOTGUN_DATA), "raw", "linear", None, [1, 1]).cache()


def get_data_tuning(self):
    return parse(self.sc.textFile(FISH_DATA), "dff").cache()


# For now, this only tests that the jobs run without crashing:
class RegressionTestCase(PySparkTestCase):
    def setUp(self):
        super(RegressionTestCase, self).setUp()
        self.outputDir = tempfile.mkdtemp()

    def tearDown(self):
        super(RegressionTestCase, self).tearDown()
        shutil.rmtree(self.outputDir)


class TestRegression(RegressionTestCase):

    def test_linear_regression(self):
        data = get_data_regression(self)
        betas, stats, comps, latent, scores, traj, r = regress(data, FISH_LINEAR_MODEL, "linear")
        stats.collect()
        scores.collect()
        r.collect()

    def test_linear_shuffle_regression(self):
        data = get_data_regression(self)
        betas, stats, comps, latent, scores, traj, r = regress(data, FISH_LINEAR_MODEL, "linear-shuffle")
        stats.collect()
        scores.collect()
        r.collect()

    def test_bilinear_regression(self):
        data = get_data_regression(self)
        betas, stats, comps, latent, scores, traj, r = regress(data, FISH_BILINEAR_MODEL, "bilinear")
        stats.collect()
        scores.collect()
        r.collect()

    def test_mean_regression(self):
        data = get_data_regression(self)
        betas, stats, comps, latent, scores, traj, r = regress(data, FISH_LINEAR_MODEL, "mean")
        stats.collect()
        scores.collect()
        r.collect()


class TestShotgun(RegressionTestCase):

    def test_shotgun(self):
        data = get_data_shotgun(self)
        b = shotgun(data, SHOTGUN_MODEL, 10)


class TestTuning(RegressionTestCase):

    def test_circular_tuning(self):
        data = get_data_tuning(self)
        params, stats, r, comps, latent, scores = tuning(data, FISH_BILINEAR_MODEL, "bilinear", "circular")
        params.collect()
        stats.collect()
        r.collect()
        scores.collect()

    def test_gaussian_tuning(self):
        data = get_data_tuning(self)
        params, stats, r, comps, latent, scores = tuning(data, FISH_BILINEAR_MODEL, "bilinear", "gaussian")
        params.collect()
        stats.collect()
        r.collect()
        scores.collect()

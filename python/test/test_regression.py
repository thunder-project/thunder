import os
import shutil
import tempfile
from thunder.regression.regress import regress
from thunder.regression.shotgun import shotgun
from thunder.regression.tuning import tuning
from test_utils import PySparkTestCase

# Hack to find the data files:
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
FISH_DATA = os.path.join(DATA_DIR, "fish.txt")
SHOTGUN_DATA = os.path.join(DATA_DIR, "shotgun.txt")
SHOTGUN_MODEL = os.path.join(DATA_DIR, "regression/shotgun")
FISH_LINEAR_MODEL = os.path.join(DATA_DIR, "regression/fish_linear")
FISH_BILINEAR_MODEL = os.path.join(DATA_DIR, "regression/fish_bilinear")


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
        regress(self.sc, FISH_DATA, FISH_LINEAR_MODEL, self.outputDir, "linear")

    def test_linear_shuffle_regression(self):
        regress(self.sc, FISH_DATA, FISH_LINEAR_MODEL, self.outputDir, "linear-shuffle")

    def test_bilinear_regression(self):
        regress(self.sc, FISH_DATA, FISH_BILINEAR_MODEL, self.outputDir, "bilinear")

    def test_mean_regression(self):
        regress(self.sc, FISH_DATA, FISH_LINEAR_MODEL, self.outputDir, "mean")


class TestShotgun(RegressionTestCase):

    def test_shotgun(self):
        shotgun(self.sc, SHOTGUN_DATA, SHOTGUN_MODEL, self.outputDir, 10)


class TestTuning(RegressionTestCase):

    def test_circular_tuning(self):
        tuning(self.sc, FISH_DATA, FISH_BILINEAR_MODEL, self.outputDir, "bilinear", "circular")

    def test_gaussian_tuning(self):
        tuning(self.sc, FISH_DATA, FISH_BILINEAR_MODEL, self.outputDir, "bilinear", "gaussian")

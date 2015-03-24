import shutil
import tempfile
import unittest
from numpy import vstack
from pyspark import SparkContext


class PySparkTestCase(unittest.TestCase):
    def setUp(self):
        class_name = self.__class__.__name__
        self.sc = SparkContext('local', class_name)
        self.sc._jvm.System.setProperty("spark.ui.showConsoleProgress", "false")
        log4j = self.sc._jvm.org.apache.log4j
        log4j.LogManager.getRootLogger().setLevel(log4j.Level.FATAL)

    def tearDown(self):
        self.sc.stop()
        # To avoid Akka rebinding to the same port, since it doesn't unbind
        # immediately on shutdown
        self.sc._jvm.System.clearProperty("spark.driver.port")


class PySparkTestCaseWithOutputDir(PySparkTestCase):
    def setUp(self):
        super(PySparkTestCaseWithOutputDir, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(PySparkTestCaseWithOutputDir, self).tearDown()
        shutil.rmtree(self.outputdir)


def elementwiseMean(arys):
    from numpy import mean
    combined = vstack([ary.ravel() for ary in arys])
    meanAry = mean(combined, axis=0)
    return meanAry.reshape(arys[0].shape)


def elementwiseVar(arys):
    from numpy import var
    combined = vstack([ary.ravel() for ary in arys])
    varAry = var(combined, axis=0)
    return varAry.reshape(arys[0].shape)


def elementwiseStdev(arys):
    from numpy import std
    combined = vstack([ary.ravel() for ary in arys])
    stdAry = std(combined, axis=0)
    return stdAry.reshape(arys[0].shape)



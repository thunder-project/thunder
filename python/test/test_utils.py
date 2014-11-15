import shutil
import tempfile
import unittest
import logging
from numpy import vstack
from pyspark import SparkContext


class PySparkTestCase(unittest.TestCase):
    def setUp(self):
        class_name = self.__class__.__name__
        self.sc = SparkContext('local', class_name)
        logging.getLogger("py4j").setLevel(logging.WARNING)
        logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)

    def tearDown(self):
        self.sc.stop()
        # To avoid Akka rebinding to the same port, since it doesn't unbind
        # immediately on shutdown
        self.sc._jvm.System.clearProperty("spark.driver.port")


class PySparkTestCaseWithOutputDir(PySparkTestCase):
    def setUp(self):
        super(PySparkTestCaseWithOutputDir, self).setUp()
        self.outputdir = tempfile.mkdtemp()
        logging.getLogger("py4j").setLevel(logging.WARNING)
        logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)

    def tearDown(self):
        super(PySparkTestCaseWithOutputDir, self).tearDown()
        shutil.rmtree(self.outputdir)


def elementwise_mean(arys):
    from numpy import mean
    combined = vstack([ary.ravel() for ary in arys])
    meanary = mean(combined, axis=0)
    return meanary.reshape(arys[0].shape)


def elementwise_var(arys):
    from numpy import var
    combined = vstack([ary.ravel() for ary in arys])
    meanary = var(combined, axis=0)
    return meanary.reshape(arys[0].shape)


def elementwise_stdev(arys):
    from numpy import std
    combined = vstack([ary.ravel() for ary in arys])
    stdary = std(combined, axis=0)
    return stdary.reshape(arys[0].shape)

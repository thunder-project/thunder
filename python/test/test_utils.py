import shutil
import tempfile
import unittest
import logging
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
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


class TestSerializableDecorator(PySparkTestCase):

    def test_serializable_decorator(self):

        from thunder.utils.decorators import serializable
        import numpy as np
        import datetime

        @serializable
        class Visitor(object):
            def __init__(self, ip_addr = None, agent = None, referrer = None):
                self.ip = ip_addr
                self.ua = agent
                self.referrer= referrer
                self.test_dict = {'a': 10, 'b': "string", 'c': [1, 2, 3]}
                self.test_vec = np.array([1,2,3])
                self.test_array = np.array([[1,2,3],[4,5,6.]])
                self.time = datetime.datetime.now()

            def __str__(self):
                return str(self.ip) + " " + str(self.ua) + " " + str(self.referrer) + " " + str(self.time)

            def test_method(self):
                return True

        # Run the test.  Build an object, serialize it, and recover it.

        # Create a new object
        orig_visitor = Visitor('192.168', 'UA-1', 'http://www.google.com')

        # Serialize the object
        pickled_visitor = orig_visitor.serialize(numpy_storage='ascii')

        # Restore object
        recov_visitor = Visitor.deserialize(pickled_visitor)

        # Check that the object was reconstructed successfully
        assert(orig_visitor.ip == recov_visitor.ip)
        assert(orig_visitor.ua == recov_visitor.ua)
        assert(orig_visitor.referrer == recov_visitor.referrer)
        for key in orig_visitor.test_dict.keys():
            assert(orig_visitor.test_dict[key] == recov_visitor.test_dict[key])

        assert(np.all(orig_visitor.test_vec == recov_visitor.test_vec))
        assert(np.all(orig_visitor.test_array == recov_visitor.test_array))

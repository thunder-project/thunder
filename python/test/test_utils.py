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

    def testSerializableDecorator(self):

        from thunder.utils.decorators import serializable
        from numpy import array, all
        from datetime import datetime

        @serializable
        class Visitor(object):
            def __init__(self, ip_addr = None, agent = None, referrer = None):
                self.ip = ip_addr
                self.ua = agent
                self.referrer= referrer
                self.testDict = {'a': 10, 'b': "string", 'c': [1, 2, 3]}
                self.testVec = array([1,2,3])
                self.testArray = array([[1,2,3],[4,5,6.]])
                self.time = datetime.now()
                self.testComplex = complex(3,2)

            def __str__(self):
                return str(self.ip) + " " + str(self.ua) + " " + str(self.referrer) + " " + str(self.time)

            def test_method(self):
                return True

        # Run the test.  Build an object, serialize it, and recover it.

        # Create a new object
        origVisitor = Visitor('192.168', 'UA-1', 'http://www.google.com')

        # Serialize the object
        pickled_visitor = origVisitor.serialize(numpyStorage='ascii')

        # Restore object
        recovVisitor = Visitor.deserialize(pickled_visitor)

        # Check that the object was reconstructed successfully
        assert(origVisitor.ip == recovVisitor.ip)
        assert(origVisitor.ua == recovVisitor.ua)
        assert(origVisitor.referrer == recovVisitor.referrer)
        assert(origVisitor.testComplex == recovVisitor.testComplex)
        for key in origVisitor.testDict.keys():
            assert(origVisitor.testDict[key] == recovVisitor.testDict[key])

        assert(all(origVisitor.testVec == recovVisitor.testVec))
        assert(all(origVisitor.testArray == recovVisitor.testArray))

    def testNotSerializable(self):
        '''
        Unit test to make sure exceptions are thrown if the object contains an
        unserializable data type.
        '''

        from thunder.utils.decorators import serializable
        from numpy import array, all
        from datetime import datetime

        class SomeOtherClass(object):
            def __init__(self):
                someVariable = 3

        @serializable
        class Visitor(object):
            def __init__(self):
                self.refrerenceToUnserializableClass = [ SomeOtherClass() ]

        origVisitor = Visitor()

        # Serialize the object
        try:
            pickled_visitor = origVisitor.serialize()   # This should fail
            assert(False)   # The @serializable wrapped class should have thrown an exception, but didn't!
        except(TypeError):
            pass            # If the exception was thrown and caught, the test has passed




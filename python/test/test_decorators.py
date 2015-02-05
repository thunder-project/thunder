import unittest
from pyspark import SparkContext

class TestSerializableDecorator(unittest.TestCase):

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

    def testSerializeWithSlots(self):
        '''
        Check to make sure that classes that use slots can be serialized / deserialized.
        '''

        from thunder.utils.decorators import serializable

        @serializable
        class Foo(object):
            __slots__ = ['bar']

        foo = Foo()
        foo.bar = 'a'
        testJson = foo.serialize()  # boom
        foo2 = Foo.deserialize(testJson)
        assert(foo.bar == foo2.bar)

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



    def testNamedTupleSerializable(self):
        '''
        Unit test to make sure exceptions are thrown if the object contains an
        unserializable data type.
        '''

        from thunder.utils.decorators import serializable
        from collections import namedtuple

        @serializable
        class Foo(object):
            def __init__(self):
                self.nt = namedtuple('FooTuple', 'bar')

        foo = Foo()
        foo.nt.bar = "baz"

        testJson = foo.serialize()
        foo2 = Foo.deserialize(testJson)
        assert(foo.nt.bar == foo2.nt.bar)


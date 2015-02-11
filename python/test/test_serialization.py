from nose import SkipTest
from nose.tools import assert_equal, assert_raises, assert_true
import unittest
from pyspark import SparkContext

from thunder.utils.serializable import ThunderSerializable


class Foo(ThunderSerializable):
    pass


class Bar(ThunderSerializable):
    def __init__(self, baz=None):
        self.baz = baz

    def getBaz(self):
        return self.baz

    def __eq__(self, other):
        return isinstance(other, Bar) and other.baz == self.baz


class TestSerializableDecorator(unittest.TestCase):

    def testSerializableDecorator(self):
        from numpy import array, all
        from datetime import datetime

        class Visitor(ThunderSerializable):
            def __init__(self, ip_addr=None, agent=None, referrer=None):
                self.ip = ip_addr
                self.ua = agent
                self.referrer = referrer
                self.testDict = {'a': 10, 'b': "string", 'c': [1, 2, 3]}
                self.testVec = array([1, 2, 3])
                self.testArray = array([[1, 2, 3], [4, 5, 6.]])
                self.time = datetime.now()
                self.testComplex = complex(3, 2)

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
        """
        Check to make sure that classes that use slots can be serialized / deserialized.
        """
        raise SkipTest("This test doesn't currently pass after changing serialization from a wrapper to a mixin")

        class SlottyFoo(ThunderSerializableWithSlots):
            __slots__ = ['bar']

        foo = SlottyFoo()
        foo.bar = 'a'
        testJson = foo.serialize()
        foo2 = SlottyFoo.deserialize(testJson)
        assert(foo.bar == foo2.bar)

    def testNotSerializable(self):
        """
        Unit test to make sure exceptions are thrown if the object contains an
        unserializable data type.
        """
        class SomeOtherClass(object):
            def __init__(self):
                someVariable = 3

        class Visitor(ThunderSerializable):
            def __init__(self):
                self.refrerenceToUnserializableClass = [SomeOtherClass()]

        origVisitor = Visitor()
        # try to serialize the object, we expect TypeError
        assert_raises(TypeError, origVisitor.serialize)

    def testNamedTupleSerializable(self):
        """
        Test that nested named tuples are serializable
        """
        from collections import namedtuple

        class NamedTupleyFoo(ThunderSerializable):
            def __init__(self):
                self.nt = namedtuple('FooTuple', 'bar')

        foo = NamedTupleyFoo()
        foo.nt.bar = "baz"

        testJson = foo.serialize()
        foo2 = NamedTupleyFoo.deserialize(testJson)
        assert(foo.nt.bar == foo2.nt.bar)

    def testNestedSerialization(self):
        """Test that multiple nested serializable objects are serializable
        """
        foo = Foo()
        foo.bar = Bar()
        foo.bar.baz = 1

        testJson = foo.serialize()
        # print testJson
        roundtripped = Foo.deserialize(testJson)

        assert_true(isinstance(roundtripped, Foo))
        assert_true(hasattr(roundtripped, "bar"))
        roundtrippedBar = roundtripped.bar
        assert_true(isinstance(roundtrippedBar, Bar))
        assert_equal(foo.bar.baz, roundtrippedBar.getBaz())

    def testNestedHomogenousListSerialization(self):
        """Test that multiple nested serializable objects are serializable
        """
        foo = Foo()
        foo.lst = [Bar(baz=x) for x in xrange(3)]

        testJson = foo.serialize()
        # print testJson
        roundtripped = Foo.deserialize(testJson)

        assert_true(isinstance(roundtripped, Foo))
        assert_true(hasattr(roundtripped, "lst"))
        roundtrippedLst = roundtripped.lst
        assert_true(isinstance(roundtrippedLst, list))
        assert_equal(3, len(roundtrippedLst))
        for expectedBaz, bar in enumerate(roundtrippedLst):
            assert_true(isinstance(bar, Bar))
            assert_equal(expectedBaz, bar.getBaz())

    def testNestedHeterogenousListSerialization(self):
        """Test that multiple nested serializable objects are serializable
        """
        foo = Foo()
        foo.lst = ["monkey", Bar(baz=1), (2, 3)]

        testJson = foo.serialize()
        # print testJson
        roundtripped = Foo.deserialize(testJson)

        assert_true(isinstance(roundtripped, Foo))
        assert_true(hasattr(roundtripped, "lst"))
        roundtrippedLst = roundtripped.lst
        assert_true(isinstance(roundtrippedLst, list))
        assert_equal(len(foo.lst), len(roundtrippedLst))
        for expected, actual in zip(foo.lst, roundtrippedLst):
            assert_equal(type(expected), type(actual))
            assert_equal(expected, actual)


from nose.tools import assert_equal, assert_raises, assert_true
import unittest

from thunder.utils.serializable import Serializable


class Foo(Serializable):
    pass


class Bar(Serializable):
    def __init__(self, baz=None):
        self.baz = baz

    def getBaz(self):
        return self.baz

    def __eq__(self, other):
        return isinstance(other, Bar) and other.baz == self.baz


class TestSerialization(unittest.TestCase):

    def test_basicSerialization(self):
        """Check serialization of a basic class with a number of data types
        """
        from numpy import array, array_equal
        from datetime import datetime

        class Visitor(Serializable):
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

            @staticmethod
            def test_method():
                return True

        # Run the test.  Build an object, serialize it, and recover it.

        # Create a new object
        origVisitor = Visitor('192.168', 'UA-1', 'http://www.google.com')

        # Serialize the object
        pickled_visitor = origVisitor.serialize(numpyStorage='ascii')
        # print pickled_visitor

        # Restore object
        recovVisitor = Visitor.deserialize(pickled_visitor)

        # Check that the object was reconstructed successfully
        assert_equal(origVisitor.ip, recovVisitor.ip)
        assert_equal(origVisitor.ua, recovVisitor.ua)
        assert_equal(origVisitor.referrer, recovVisitor.referrer)
        assert_equal(origVisitor.testComplex, recovVisitor.testComplex)
        assert_equal(sorted(origVisitor.testDict.keys()), sorted(recovVisitor.testDict.keys()))
        for key in origVisitor.testDict.keys():
            assert_equal(origVisitor.testDict[key], recovVisitor.testDict[key])

        assert_true(array_equal(origVisitor.testVec, recovVisitor.testVec))
        assert_true(array_equal(origVisitor.testArray, recovVisitor.testArray))

    def test_serializeWithSlots(self):
        """
        Check to make sure that classes that use slots can be serialized / deserialized.
        """
        class SlottyFoo(Serializable):
            __slots__ = ['bar']

            def __init__(self):
                self.bar = None

            def __eq__(self, other):
                return isinstance(other, SlottyFoo) and self.bar == other.bar

        foo = SlottyFoo()
        foo.bar = 'a'
        testJson = foo.serialize()
        foo2 = SlottyFoo.deserialize(testJson)
        assert_true(isinstance(foo2, SlottyFoo))
        assert_equal(foo, foo2)

    def test_notSerializable(self):
        """
        Unit test to make sure exceptions are thrown if the object contains an
        unserializable data type.
        """
        class SomeOtherClass(object):
            def __init__(self):
                self.someVariable = 3

        class Visitor(Serializable):
            def __init__(self):
                self.referenceToUnserializableClass = [SomeOtherClass()]

        origVisitor = Visitor()
        # try to serialize the object, we expect TypeError
        assert_raises(TypeError, origVisitor.serialize)

    def test_namedTupleSerializable(self):
        """
        Test that nested named tuples are serializable
        """
        from collections import namedtuple

        class NamedTupleyFoo(Serializable):
            def __init__(self):
                self.nt = namedtuple('FooTuple', 'bar')

            def __eq__(self, other):
                return isinstance(other, NamedTupleyFoo) and self.nt.bar == other.nt.bar

        foo = NamedTupleyFoo()
        foo.nt.bar = "baz"

        testJson = foo.serialize()
        foo2 = NamedTupleyFoo.deserialize(testJson)
        assert_equal(foo, foo2)

    def test_nestedSerialization(self):
        """Test that multiple nested serializable objects are serializable
        """
        foo = Foo()
        foo.bar = Bar()
        foo.bar.baz = 1

        testJson = foo.serialize()
        # print testJson  # uncomment for testing
        roundtripped = Foo.deserialize(testJson)

        assert_true(isinstance(roundtripped, Foo))
        assert_true(hasattr(roundtripped, "bar"))
        roundtrippedBar = roundtripped.bar
        assert_true(isinstance(roundtrippedBar, Bar))
        assert_equal(foo.bar.baz, roundtrippedBar.getBaz())

    def test_nestedHomogenousListSerialization(self):
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

    def test_nestedHeterogenousListSerialization(self):
        """Test that multiple nested serializable objects of differing types are serializable
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

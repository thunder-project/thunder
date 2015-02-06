from nose.tools import assert_equals, assert_is_none, assert_raises, assert_true
from numpy import array, array_equal

from thunder.rdds.data import Data
from test_utils import PySparkTestCase


class TestImagesGetters(PySparkTestCase):
    """Test `get` and related methods on an Images-like Data object
    """
    def setUp(self):
        super(TestImagesGetters, self).setUp()
        self.ary1 = array([[1, 2], [3, 4]], dtype='int16')
        self.ary2 = array([[5, 6], [7, 8]], dtype='int16')
        rdd = self.sc.parallelize([(0, self.ary1), (1, self.ary2)])
        self.images = Data(rdd, dtype='int16')

    def test_getMissing(self):
        assert_is_none(self.images.get(-1))

    def test_get(self):
        assert_true(array_equal(self.ary2, self.images.get(1)))

        # keys are integers, ask for sequence
        assert_raises(ValueError, self.images.get, (1, 2))

    def test_getMany(self):
        vals = self.images.getMany([0, -1, 1, 0])
        assert_equals(4, len(vals))
        assert_true(array_equal(self.ary1, vals[0]))
        assert_is_none(vals[1])
        assert_true(array_equal(self.ary2, vals[2]))
        assert_true(array_equal(self.ary1, vals[3]))

        # keys are integers, ask for sequences:
        assert_raises(ValueError, self.images.get, [(0, 0)])
        assert_raises(ValueError, self.images.get, [0, (0, 0), 1, 0])

    def test_getRanges(self):
        vals = self.images.getRange(slice(None))
        assert_equals(2, len(vals))
        assert_equals(0, vals[0][0])
        assert_equals(1, vals[1][0])
        assert_true(array_equal(self.ary1, vals[0][1]))
        assert_true(array_equal(self.ary2, vals[1][1]))

        vals = self.images.getRange(slice(0, 1))
        assert_equals(1, len(vals))
        assert_equals(0, vals[0][0])
        assert_true(array_equal(self.ary1, vals[0][1]))

        vals = self.images.getRange(slice(1))
        assert_equals(1, len(vals))
        assert_equals(0, vals[0][0])
        assert_true(array_equal(self.ary1, vals[0][1]))

        vals = self.images.getRange(slice(1, 2))
        assert_equals(1, len(vals))
        assert_equals(1, vals[0][0])
        assert_true(array_equal(self.ary2, vals[0][1]))

        vals = self.images.getRange(slice(2, 3))
        assert_equals(0, len(vals))

        # keys are integers, ask for sequence
        assert_raises(ValueError, self.images.getRange, [slice(1), slice(1)])

        # raise exception if 'step' specified:
        assert_raises(ValueError, self.images.getRange, slice(1, 2, 2))

    def test_brackets(self):
        vals = self.images[1]
        assert_true(array_equal(self.ary2, vals))

        vals = self.images[0:1]
        assert_equals(1, len(vals))
        assert_equals(0, vals[0][0])
        assert_true(array_equal(self.ary1, vals[0][1]))

        vals = self.images[:]
        assert_equals(2, len(vals))
        assert_equals(0, vals[0][0])
        assert_equals(1, vals[1][0])
        assert_true(array_equal(self.ary1, vals[0][1]))
        assert_true(array_equal(self.ary2, vals[1][1]))

        vals = self.images[1:4]
        assert_equals(1, len(vals))
        assert_equals(1, vals[0][0])
        assert_true(array_equal(self.ary2, vals[0][1]))

        vals = self.images[1:]
        assert_equals(1, len(vals))
        assert_equals(1, vals[0][0])
        assert_true(array_equal(self.ary2, vals[0][1]))

        vals = self.images[:1]
        assert_equals(1, len(vals))
        assert_equals(0, vals[0][0])
        assert_true(array_equal(self.ary1, vals[0][1]))

        assert_raises(KeyError, self.images.__getitem__, 2)  # equiv: self.images[2]

        assert_raises(IndexError, self.images.__getitem__, slice(2, 3))  # equiv: self.images[2:3]


class TestSeriesGetters(PySparkTestCase):
    """Test `get` and related methods on a Series-like Data object
    """
    def setUp(self):
        super(TestSeriesGetters, self).setUp()
        self.dataLocal = [
            ((0, 0), array([1.0, 2.0, 3.0], dtype='float32')),
            ((0, 1), array([2.0, 2.0, 4.0], dtype='float32')),
            ((1, 0), array([4.0, 2.0, 1.0], dtype='float32')),
            ((1, 1), array([3.0, 1.0, 1.0], dtype='float32'))
        ]
        self.series = Data(self.sc.parallelize(self.dataLocal), dtype='float32')

    def test_getMissing(self):
        assert_is_none(self.series.get((-1, -1)))

    def test_get(self):
        expected = self.dataLocal[1][1]
        assert_true(array_equal(expected, self.series.get((0, 1))))

        assert_raises(ValueError, self.series.get, 1)  # keys are sequences, ask for integer
        assert_raises(ValueError, self.series.get, (1, 2, 3))  # key length mismatch

    def test_getMany(self):
        vals = self.series.getMany([(0, 0), (17, 256), (1, 0), (0, 0)])
        assert_equals(4, len(vals))
        assert_true(array_equal(self.dataLocal[0][1], vals[0]))
        assert_is_none(vals[1])
        assert_true(array_equal(self.dataLocal[2][1], vals[2]))
        assert_true(array_equal(self.dataLocal[0][1], vals[3]))

        assert_raises(ValueError, self.series.getMany, [1])  # keys are sequences, ask for integer
        assert_raises(ValueError, self.series.getMany, [(0, 0), 1, (1, 0), (0, 0)])  # asking for integer again

    def test_getRanges(self):
        vals = self.series.getRange([slice(2), slice(2)])
        assert_equals(4, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[1][0], vals[1][0])
        assert_equals(self.dataLocal[2][0], vals[2][0])
        assert_equals(self.dataLocal[3][0], vals[3][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1][1]))
        assert_true(array_equal(self.dataLocal[2][1], vals[2][1]))
        assert_true(array_equal(self.dataLocal[3][1], vals[3][1]))

        vals = self.series.getRange([slice(2), slice(1)])
        assert_equals(2, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[2][0], vals[1][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[2][1], vals[1][1]))

        vals = self.series.getRange([slice(None), slice(1, 2)])
        assert_equals(2, len(vals))
        assert_equals(self.dataLocal[1][0], vals[0][0])
        assert_equals(self.dataLocal[3][0], vals[1][0])
        assert_true(array_equal(self.dataLocal[1][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[3][1], vals[1][1]))

        vals = self.series.getRange([slice(None), slice(None)])
        assert_equals(4, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[1][0], vals[1][0])
        assert_equals(self.dataLocal[2][0], vals[2][0])
        assert_equals(self.dataLocal[3][0], vals[3][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1][1]))
        assert_true(array_equal(self.dataLocal[2][1], vals[2][1]))
        assert_true(array_equal(self.dataLocal[3][1], vals[3][1]))

        vals = self.series.getRange([0, slice(None)])
        assert_equals(2, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[1][0], vals[1][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1][1]))

        vals = self.series.getRange([0, 1])
        assert_equals(1, len(vals))
        assert_equals(self.dataLocal[1][0], vals[0][0])
        assert_true(array_equal(self.dataLocal[1][1], vals[0][1]))

        vals = self.series.getRange([slice(2, 3), slice(None)])
        assert_equals(0, len(vals))

        # keys are sequences, ask for single slice
        assert_raises(ValueError, self.series.getRange, slice(2, 3))

        # ask for wrong number of slices
        assert_raises(ValueError, self.series.getRange, [slice(2, 3), slice(2, 3), slice(2, 3)])

        # raise exception if 'step' specified:
        assert_raises(ValueError, self.series.getRange, [slice(0, 4, 2), slice(2, 3)])

    def test_brackets(self):
        # returns just value; calls `get`
        vals = self.series[(1, 0)]
        assert_true(array_equal(self.dataLocal[2][1], vals))

        # tuple isn't needed; returns just value, calls `get`
        vals = self.series[0, 1]
        assert_true(array_equal(self.dataLocal[1][1], vals))

        # if slices are passed, calls `getRange`, returns keys and values
        vals = self.series[0:1, 1:2]
        assert_equals(1, len(vals))
        assert_equals(self.dataLocal[1][0], vals[0][0])
        assert_true(array_equal(self.dataLocal[1][1], vals[0][1]))

        # if slice extends out of bounds, return only the elements that are in bounds
        vals = self.series[:4, :1]
        assert_equals(2, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[2][0], vals[1][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[2][1], vals[1][1]))

        # empty slice works
        vals = self.series[:, 1:2]
        assert_equals(2, len(vals))
        assert_equals(self.dataLocal[1][0], vals[0][0])
        assert_equals(self.dataLocal[3][0], vals[1][0])
        assert_true(array_equal(self.dataLocal[1][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[3][1], vals[1][1]))

        # multiple empty slices work
        vals = self.series[:, :]
        assert_equals(4, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[1][0], vals[1][0])
        assert_equals(self.dataLocal[2][0], vals[2][0])
        assert_equals(self.dataLocal[3][0], vals[3][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1][1]))
        assert_true(array_equal(self.dataLocal[2][1], vals[2][1]))
        assert_true(array_equal(self.dataLocal[3][1], vals[3][1]))

        # mixing slices and individual indicies works:
        vals = self.series[0, :]
        assert_equals(2, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[1][0], vals[1][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1][1]))

        # trying to getitem a key that doesn't exist raises KeyError
        # this differs from `get` behavior but is consistent with python dict
        # see object.__getitem__ in https://docs.python.org/2/reference/datamodel.html
        assert_raises(KeyError, self.series.__getitem__, (25, 17))  # equiv: self.series[(25, 17)]

        # passing a range that is completely out of bounds throws IndexError
        # note that if a range is only partly out of bounds, it will return what elements the slice does include
        assert_raises(IndexError, self.series.__getitem__, [slice(2, 3), slice(None)])  # series[2:3,:]

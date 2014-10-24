import itertools
from numpy import arange, array, array_equal, concatenate, dtype, prod
import unittest
from nose.tools import assert_equals, assert_true, assert_almost_equal, assert_raises
from thunder.rdds.imageblocks import ImageBlockValue


class TestImageBlockValue(unittest.TestCase):

    def test_fromArrayByPlane(self):
        values = arange(12, dtype='int16').reshape((3, 4), order='C')

        planedim = 0
        planedimidx = 1
        imageblock = ImageBlockValue.fromArrayByPlane(values, planedim=planedim, planeidx=planedimidx)

        assert_equals(values.shape, imageblock.origshape)
        assert_equals(slice(planedimidx, planedimidx+1, 1), imageblock.origslices[planedim])
        assert_equals(slice(None), imageblock.origslices[1])
        assert_true(array_equal(values[planedimidx, :].flatten(order='C'), imageblock.values.flatten(order='C')))

    def test_fromArrayBySlices(self):
        values = arange(12, dtype='int16').reshape((3, 4), order='C')

        slices = [[slice(0, 3)], [slice(0, 2), slice(2, 4)]]
        slicesiter = itertools.product(*slices)

        imageblocks = [ImageBlockValue.fromArrayBySlices(values, sls) for sls in slicesiter]
        assert_equals(2, len(imageblocks))
        assert_equals((3, 2), imageblocks[0].values.shape)
        assert_true(array_equal(values[(slice(0, 3), slice(0, 2))], imageblocks[0].values))

    def test_fromPlanarBlocks(self):
        values = arange(36, dtype='int16').reshape((3, 4, 3), order='F')

        imageblocks = [ImageBlockValue.fromArrayByPlane(values, -1, i) for i in xrange(values.shape[2])]

        recombblock = ImageBlockValue.fromPlanarBlocks(imageblocks, planarDim=-1)

        assert_true(array_equal(values, recombblock.values))
        assert_equals([slice(None)] * values.ndim, recombblock.origslices)
        assert_equals(values.shape, recombblock.origshape)

    def test_addDimension(self):
        values = arange(12, dtype='int16').reshape((3, 4), order='C')
        morevalues = arange(12, 24, dtype='int16').reshape((3, 4), order='C')

        origshape = values.shape
        origslices = [slice(None)] * values.ndim
        newdimsize = 2
        initimageblock = ImageBlockValue(origshape=origshape, origslices=origslices, values=values)
        anotherinitimageblock = ImageBlockValue(origshape=origshape, origslices=origslices, values=morevalues)

        imageblock = initimageblock.addDimension(newdimidx=0, newdimsize=newdimsize)
        anotherimageblock = anotherinitimageblock.addDimension(newdimidx=1, newdimsize=newdimsize)

        expectedorigshape = tuple([newdimsize] + list(initimageblock.origshape))
        assert_equals(expectedorigshape, imageblock.origshape)
        assert_equals(expectedorigshape, anotherimageblock.origshape)

        expectednslices = len(expectedorigshape)
        assert_equals(expectednslices, len(imageblock.origslices))
        assert_equals(expectednslices, len(anotherimageblock.origslices))

        assert_equals(slice(0, 1, 1), imageblock.origslices[0])
        assert_equals(slice(1, 2, 1), anotherimageblock.origslices[0])

        expectedshape = tuple([1] + list(values.shape))
        assert_equals(expectedshape, imageblock.values.shape)
        assert_equals(expectedshape, anotherimageblock.values.shape)

        # check that straight array concatenation works as expected in this particular case
        expectedcatvals = arange(24, dtype='int16')
        actualcatvals = concatenate((imageblock.values, anotherimageblock.values), axis=0).flatten(order='C')
        assert_true(array_equal(expectedcatvals, actualcatvals))

    def test_toSeriesIter(self):
        sh = 2, 3, 4
        sz = prod(sh)
        ary = arange(sz, dtype=dtype('int16')).reshape(sh)
        imageblock = ImageBlockValue.fromArray(ary)

        series = list(imageblock.toSeriesIter(-1))

        expectedseries = [
            ((0, 0), array([0, 1, 2, 3], dtype='int16')),
            ((1, 0), array([4, 5, 6, 7], dtype='int16')),
            ((2, 0), array([8, 9, 10, 11], dtype='int16')),
            ((0, 1), array([12, 13, 14, 15], dtype='int16')),
            ((1, 1), array([16, 17, 18, 19], dtype='int16')),
            ((2, 1), array([20, 21, 22, 23], dtype='int16')),
        ]

        for actual, expected in zip(series, expectedseries):
            # check key equality
            assert_equals(expected[0], actual[0])
            # check value equality
            assert_true(array_equal(expected[1], actual[1]))

    def test_toSeriesIter2(self):
        # add singleton dimension on end of (2, 4) shape to be "time":
        ary = arange(8, dtype=dtype('int16')).reshape((2, 4, 1))

        imageblock = ImageBlockValue.fromArray(ary)

        seriesvals = list(imageblock.toSeriesIter(-1))

        # check ordering of keys
        assert_equals((0, 0), seriesvals[0][0])  # first key
        assert_equals((1, 0), seriesvals[1][0])  # second key
        assert_equals((0, 1), seriesvals[2][0])
        assert_equals((1, 1), seriesvals[3][0])
        assert_equals((0, 2), seriesvals[4][0])
        assert_equals((1, 2), seriesvals[5][0])
        assert_equals((0, 3), seriesvals[6][0])
        assert_equals((1, 3), seriesvals[7][0])

        # check that values are in original order
        collectedvals = array([kv[1] for kv in seriesvals], dtype=dtype('int16')).ravel()
        assert_true(array_equal(ary.ravel(order='F'), collectedvals))

from collections import namedtuple
from nose.tools import assert_equals, assert_true
from numpy import arange, array_equal, expand_dims, isclose
import unittest

from thunder.rdds.imgblocks.strategy import PaddedBlockingStrategy, SimpleBlockingStrategy


class TestSimpleSplitCalculation(unittest.TestCase):

    @staticmethod
    def _run_tst_splitCalc(blockSize, image, expectedSplits, expectedSize, testIdx=-1):
        strategy = SimpleBlockingStrategy.generateFromBlockSize(image, blockSize)
        splits = strategy.splitsPerDim
        avgSize = strategy.calcAverageBlockSize()
        assert_equals(tuple(expectedSplits), tuple(splits),
                      msg="Failure in test %i, expected splits %s != actual splits %s" %
                          (testIdx, tuple(expectedSplits), tuple(splits)))
        assert_true(isclose(expectedSize, avgSize, rtol=0.001),
                    msg="Failure in test %i, expected avg size %g not close to actual size %g" %
                        (testIdx, expectedSize, avgSize))

    def test_splitCalc(self):
        MockImage = namedtuple("MockImage", "dims nimages dtype")
        PARAMS = [
            (1, MockImage((2, 2, 2), 1, "uint8"), (2, 2, 2), 1),
            (2, MockImage((2, 2, 2), 2, "uint8"), (2, 2, 2), 2),
            (2, MockImage((2, 2, 2), 1, "uint16"), (2, 2, 2), 2),
            (800000, MockImage((200, 200, 30), 5, "uint32"), (1, 1, 30), 800000),
            ("150MB", MockImage((2048, 1060, 36), 1000, "uint8"), (1, 14, 36), 1.55e+08)]
        for testIdx, params in enumerate(PARAMS):
            TestSimpleSplitCalculation._run_tst_splitCalc(*params, testIdx=testIdx)


class TestBlockExtraction(unittest.TestCase):
    ExtractParams = namedtuple("ExtractParams", "aryshape blockslices timepoint ntimepoints padding")
    PARAMS = [ExtractParams((2, 2), (slice(None), slice(0, 1)), 5, 10, 0),
              ExtractParams((12, 12), (slice(3, 6, 1), slice(6, 9, 1)), 5, 10, 0),
              ExtractParams((12, 12), (slice(3, 6, 1), slice(6, 9, 1)), 5, 10, 2)]

    def test_simpleBlockExtraction(self):
        for params in TestBlockExtraction.PARAMS:
            strategy = SimpleBlockingStrategy([1]*len(params.aryshape))  # dummy splits; not used here
            n = reduce(lambda x, y: x*y, params.aryshape)
            ary = arange(n, dtype='int16').reshape(params.aryshape)
            key, val = strategy.extractBlockFromImage(ary, params.blockslices, params.timepoint, params.ntimepoints)

            expectedSlices = [slice(params.timepoint, params.timepoint+1, 1)] + list(params.blockslices)
            expectedAry = expand_dims(ary[params.blockslices], axis=0)
            assert_equals(params.timepoint, key.temporalKey)
            assert_equals(params.ntimepoints, key.origshape[0])
            assert_equals(tuple(params.aryshape), tuple(key.origshape[1:]))
            assert_equals(tuple(expectedSlices), tuple(key.imgslices))
            assert_true(array_equal(expectedAry, val))

    def test_paddedBlockExtraction(self):
        for params in TestBlockExtraction.PARAMS:
            strategy = PaddedBlockingStrategy([1]*len(params.aryshape), params.padding)  # dummy splits; not used here
            n = reduce(lambda x, y: x*y, params.aryshape)
            ary = arange(n, dtype='int16').reshape(params.aryshape)
            key, val = strategy.extractBlockFromImage(ary, params.blockslices, params.timepoint, params.ntimepoints)

            expectedSlices = [slice(params.timepoint, params.timepoint+1, 1)] + list(params.blockslices)
            assert_equals(params.timepoint, key.temporalKey)
            assert_equals(params.ntimepoints, key.origshape[0])
            assert_equals(tuple(params.aryshape), tuple(key.origshape[1:]))
            assert_equals(tuple(expectedSlices), tuple(key.imgslices))

            try:
                _ = len(params.padding)
                padding = list(params.padding)
            except TypeError:
                padding = [params.padding] * ary.ndim

            expectedPaddedSlices = []
            expectedValSlices = []
            for slise, pad, l in zip(params.blockslices, padding, ary.shape):
                paddedStart = max(0, slise.start - pad) if not (slise.start is None) else 0
                paddedEnd = min(l, slise.stop + pad) if not (slise.stop is None) else l
                actualPadStart = slise.start - paddedStart if not (slise.start is None) else 0
                actualPadEnd = paddedEnd - slise.stop if not (slise.stop is None) else 0
                expectedPaddedSlices.append(slice(paddedStart, paddedEnd, 1))
                expectedValSlices.append(slice(actualPadStart, (paddedEnd-paddedStart)-actualPadEnd, 1))

            expectedAry = expand_dims(ary[expectedPaddedSlices], axis=0)
            expectedPaddedSlices = [slice(params.timepoint, params.timepoint+1, 1)] + expectedPaddedSlices
            expectedValSlices = [slice(0, 1, 1)] + expectedValSlices
            assert_equals(tuple(expectedPaddedSlices), tuple(key.padimgslices))
            assert_equals(tuple(expectedValSlices), tuple(key.valslices))
            assert_equals(tuple(expectedAry.shape), tuple(val.shape))
            assert_true(array_equal(expectedAry, val))

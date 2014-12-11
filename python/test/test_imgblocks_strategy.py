from collections import namedtuple
from nose.tools import assert_equals, assert_true
from numpy import isclose
import unittest

from thunder.rdds.imgblocks.strategy import SimpleBlockingStrategy


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
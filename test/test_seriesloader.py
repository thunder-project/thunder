import glob
import json
import os
import struct
import unittest

from nose.tools import assert_almost_equal, assert_equals, assert_true, assert_raises
from numpy import allclose, arange, array, array_equal
from numpy import dtype as dtypeFunc

from thunder.rdds.fileio.seriesloader import SeriesLoader
from thunder.utils.common import smallestFloatType
from test_utils import PySparkTestCase, PySparkTestCaseWithOutputDir

_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    Image = None


class SeriesBinaryTestData(object):
    """
    Data object for SeriesLoader binary test.
    """
    __slots__ = ('keys', 'vals', 'keyDtype', 'valDtype')

    def __init__(self, keys, vals, keyDtype, valDtype):
        """
        Constructor, intended to be called from fromArrays class factory method.

        Expects m x n and m x p data for keys and vals.

        Parameters
        ----------
        keys: two dimensional array or sequence
        vals: two dimensional array or sequence
        keydtype: object castable to numpy dtype
            data type of keys
        valdtype: object castable to numpy dtype
            data type of values

        Returns
        -------
        self: new instance of SeriesBinaryTestData
        """
        self.keys = keys
        self.vals = vals
        self.keyDtype = keyDtype
        self.valDtype = valDtype

    @property
    def keyStructFormat(self):
        return self.keyDtype.char * self.nkeys

    @property
    def valStructFormat(self):
        return self.valDtype.char * self.nvals

    @property
    def data(self):
        return zip(self.keys, self.vals)

    @property
    def nkeys(self):
        return len(self.keys[0])

    @property
    def nvals(self):
        return len(self.vals[0])

    def writeToFile(self, f):
        """
        Writes own key, value data to passed file handle in binary format
        Parameters
        ----------
        f: file handle, open for writing
            f will remain open after this call
        """
        for keys, vals in self.data:
            f.write(struct.pack(self.keyStructFormat, *keys))
            f.write(struct.pack(self.valStructFormat, *vals))

    @staticmethod
    def _validateLengths(dat):
        l = len(dat[0])
        for d in dat:
            assert len(d) == l, "Data of unequal lengths, %d and %d" % (l, len(d))

    @staticmethod
    def _normalizeDType(dtypeInstance, data):
        if dtypeInstance is None:
            return data.dtype
        return dtypeFunc(dtypeInstance)

    @classmethod
    def fromArrays(cls, keys, vals, keyDtype=None, valDtype=None):
        """
        Factory method for SeriesBinaryTestData. Validates input before calling class __init__ method.

        Expects m x n and m x p data for keys and vals.

        Parameters
        ----------
        keys: two dimensional array or sequence
        vals: two dimensional array or sequence
        keydtype: object castable to numpy dtype
            data type of keys
        valdtype: object castable to numpy dtype
            data type of values

        Returns
        -------
        self: new instance of SeriesBinaryTestData
        """
        keyDtype = cls._normalizeDType(keyDtype, keys)
        valDtype = cls._normalizeDType(valDtype, vals)
        assert len(keys) == len(vals), "Unequal numbers of keys and values, %d and %d" % (len(keys), len(vals))
        cls._validateLengths(keys)
        cls._validateLengths(vals)
        return cls(keys, vals, keyDtype, valDtype)


class TestSeriesLoader(PySparkTestCase):
    @staticmethod
    def _findTestResourcesDir(resourcesDirName="resources"):
        testDirPath = os.path.dirname(os.path.realpath(__file__))
        testResourcesDirPath = os.path.join(testDirPath, resourcesDirName)
        if not os.path.isdir(testResourcesDirPath):
            raise IOError("Test resources directory "+testResourcesDirPath+" not found")
        return testResourcesDirPath

    @staticmethod
    def _findSourceTreeDir(dirName="utils/data"):
        testDirPath = os.path.dirname(os.path.realpath(__file__))
        testResourcesDirPath = os.path.join(testDirPath, "..", "thunder", dirName)
        if not os.path.isdir(testResourcesDirPath):
            raise IOError("Directory "+testResourcesDirPath+" not found")
        return testResourcesDirPath

    def test_fromArrays(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))

        series = SeriesLoader(self.sc).fromArraysAsImages(ary)

        seriesVals = series.collect()
        seriesAry = series.pack()

        # check ordering of keys
        assert_equals((0, 0), seriesVals[0][0])  # first key
        assert_equals((1, 0), seriesVals[1][0])  # second key
        assert_equals((2, 0), seriesVals[2][0])
        assert_equals((3, 0), seriesVals[3][0])
        assert_equals((0, 1), seriesVals[4][0])
        assert_equals((1, 1), seriesVals[5][0])
        assert_equals((2, 1), seriesVals[6][0])
        assert_equals((3, 1), seriesVals[7][0])

        # check dimensions tuple is reversed from numpy shape
        assert_equals(ary.shape[::-1], series.dims.count)

        # check that values are in original order
        collectedVals = array([kv[1] for kv in seriesVals], dtype=dtypeFunc('int16')).ravel()
        assert_true(array_equal(ary.ravel(), collectedVals))

        # check that packing returns transpose of original array
        assert_true(array_equal(ary.T, seriesAry))

    def test_fromMultipleArrays(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))
        ary2 = arange(8, 16, dtype=dtypeFunc('int16')).reshape((2, 4))

        series = SeriesLoader(self.sc).fromArraysAsImages([ary, ary2])

        seriesVals = series.collect()
        seriesAry = series.pack()

        # check ordering of keys
        assert_equals((0, 0), seriesVals[0][0])  # first key
        assert_equals((1, 0), seriesVals[1][0])  # second key
        assert_equals((3, 0), seriesVals[3][0])
        assert_equals((0, 1), seriesVals[4][0])
        assert_equals((3, 1), seriesVals[7][0])

        # check dimensions tuple is reversed from numpy shape
        assert_equals(ary.shape[::-1], series.dims.count)

        # check that values are in original order, with subsequent point concatenated in values
        collectedVals = array([kv[1] for kv in seriesVals], dtype=dtypeFunc('int16'))
        assert_true(array_equal(ary.ravel(), collectedVals[:, 0]))
        assert_true(array_equal(ary2.ravel(), collectedVals[:, 1]))

        # check that packing returns concatenation of input arrays, with time as first dimension
        assert_true(array_equal(ary.T, seriesAry[0]))
        assert_true(array_equal(ary2.T, seriesAry[1]))


class TestSeriesBinaryLoader(PySparkTestCaseWithOutputDir):

    def _run_tst_fromBinary(self, useConfJson=False):
        # run this as a single big test so as to avoid repeated setUp and tearDown of the spark context
        # data will be a sequence of test data
        # all keys and all values in a test data item must be of the same length
        # keys get converted to ints regardless of raw input format
        DATA = [
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int16'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3], [5, 6, 7]], [[11], [12]], 'int16', 'int16'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int32'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int32', 'int16'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11.0, 12.0, 13.0]], 'int16', 'float32'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11.0, 12.0, 13.0]], 'float32', 'float32'),
            SeriesBinaryTestData.fromArrays([[2, 3, 4]], [[11.0, 12.0, 13.0]], 'float32', 'float32'),
        ]

        for itemidx, item in enumerate(DATA):
            outSubdir = os.path.join(self.outputdir, 'input%d' % itemidx)
            os.mkdir(outSubdir)

            fname = os.path.join(outSubdir, 'inputfile%d.bin' % itemidx)
            with open(fname, 'wb') as f:
                item.writeToFile(f)

            loader = SeriesLoader(self.sc)
            if not useConfJson:
                series = loader.fromBinary(outSubdir, nkeys=item.nkeys, nvalues=item.nvals, keyType=str(item.keyDtype),
                                           valueType=str(item.valDtype))
            else:
                # write configuration file
                conf = {'input': outSubdir,
                        'nkeys': item.nkeys, 'nvalues': item.nvals,
                        'valuetype': str(item.valDtype), 'keytype': str(item.keyDtype)}
                with open(os.path.join(outSubdir, "conf.json"), 'wb') as f:
                    json.dump(conf, f, indent=2)
                series = loader.fromBinary(outSubdir)

            seriesData = series.rdd.collect()

            expectedData = item.data
            assert_equals(len(expectedData), len(seriesData),
                          "Differing numbers of k/v pairs in item %d; expected %d, got %d" %
                          (itemidx, len(expectedData), len(seriesData)))

            for expected, actual in zip(expectedData, seriesData):
                expectedKeys = tuple(expected[0])
                expectedType = smallestFloatType(item.valDtype)
                expectedVals = array(expected[1], dtype=expectedType)
                assert_equals(expectedKeys, actual[0],
                              "Key mismatch in item %d; expected %s, got %s" %
                              (itemidx, str(expectedKeys), str(actual[0])))
                assert_true(allclose(expectedVals, actual[1]),
                            "Value mismatch in item %d; expected %s, got %s" %
                            (itemidx, str(expectedVals), str(actual[1])))
                assert_equals(expectedType, str(actual[1].dtype),
                              "Value type mismatch in item %d; expected %s, got %s" %
                              (itemidx, expectedType, str(actual[1].dtype)))

    def test_fromBinary(self):
        self._run_tst_fromBinary()

    def test_fromBinaryWithConfFile(self):
        self._run_tst_fromBinary(True)

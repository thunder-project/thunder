import json
from numpy import allclose, arange, array, array_equal, dtype
import os
import struct
import unittest
from nose.tools import assert_equals, assert_true, assert_almost_equal

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
    __slots__ = ('keys', 'vals', 'keyDType', 'valDType')

    def __init__(self, keys, vals, keydtype, valdtype):
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
        self.keyDType = keydtype
        self.valDType = valdtype

    @property
    def keyStructFormat(self):
        return self.keyDType.char * self.nkeys

    @property
    def valStructFormat(self):
        return self.valDType.char * self.nvals

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
    def _normalizeDType(dtypeinst, data):
        if dtypeinst is None:
            return data.dtype
        return dtype(dtypeinst)

    @classmethod
    def fromArrays(cls, keys, vals, keydtype=None, valdtype=None):
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
        keydtype = cls._normalizeDType(keydtype, keys)
        valdtype = cls._normalizeDType(valdtype, vals)
        assert len(keys) == len(vals), "Unequal numbers of keys and values, %d and %d" % (len(keys), len(vals))
        cls._validateLengths(keys)
        cls._validateLengths(vals)
        return cls(keys, vals, keydtype, valdtype)


class TestSeriesLoader(PySparkTestCase):
    @staticmethod
    def _findTestResourcesDir(resourcesdirname="resources"):
        testdirpath = os.path.dirname(os.path.realpath(__file__))
        testresourcesdirpath = os.path.join(testdirpath, resourcesdirname)
        if not os.path.isdir(testresourcesdirpath):
            raise IOError("Test resources directory "+testresourcesdirpath+" not found")
        return testresourcesdirpath

    @staticmethod
    def _findSourceTreeDir(dirname="utils/data"):
        testdirpath = os.path.dirname(os.path.realpath(__file__))
        testresourcesdirpath = os.path.join(testdirpath, "..", "thunder", dirname)
        if not os.path.isdir(testresourcesdirpath):
            raise IOError("Directory "+testresourcesdirpath+" not found")
        return testresourcesdirpath

    def test_fromArrays(self):
        ary = arange(8, dtype=dtype('int16')).reshape((2, 4))

        series = SeriesLoader(self.sc).fromArrays(ary)

        seriesvals = series.collect()
        seriesary = series.pack()

        # check ordering of keys
        assert_equals((0, 0), seriesvals[0][0])  # first key
        assert_equals((1, 0), seriesvals[1][0])  # second key
        assert_equals((2, 0), seriesvals[2][0])
        assert_equals((3, 0), seriesvals[3][0])
        assert_equals((0, 1), seriesvals[4][0])
        assert_equals((1, 1), seriesvals[5][0])
        assert_equals((2, 1), seriesvals[6][0])
        assert_equals((3, 1), seriesvals[7][0])

        # check dimensions tuple is reversed from numpy shape
        assert_equals(ary.shape[::-1], series.dims.count)

        # check that values are in original order
        collectedvals = array([kv[1] for kv in seriesvals], dtype=dtype('int16')).ravel()
        assert_true(array_equal(ary.ravel(), collectedvals))

        # check that packing returns transpose of original array
        assert_true(array_equal(ary.T, seriesary))

    def test_fromMultipleArrays(self):
        ary = arange(8, dtype=dtype('int16')).reshape((2, 4))
        ary2 = arange(8, 16, dtype=dtype('int16')).reshape((2, 4))

        series = SeriesLoader(self.sc).fromArrays([ary, ary2])

        seriesvals = series.collect()
        seriesary = series.pack()

        # check ordering of keys
        assert_equals((0, 0), seriesvals[0][0])  # first key
        assert_equals((1, 0), seriesvals[1][0])  # second key
        assert_equals((3, 0), seriesvals[3][0])
        assert_equals((0, 1), seriesvals[4][0])
        assert_equals((3, 1), seriesvals[7][0])

        # check dimensions tuple is reversed from numpy shape
        assert_equals(ary.shape[::-1], series.dims.count)

        # check that values are in original order, with subsequent point concatenated in values
        collectedvals = array([kv[1] for kv in seriesvals], dtype=dtype('int16'))
        assert_true(array_equal(ary.ravel(), collectedvals[:, 0]))
        assert_true(array_equal(ary2.ravel(), collectedvals[:, 1]))

        # check that packing returns concatenation of input arrays, with time as first dimension
        assert_true(array_equal(ary.T, seriesary[0]))
        assert_true(array_equal(ary2.T, seriesary[1]))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_fromMultipageTif(self):
        testresourcesdir = TestSeriesLoader._findTestResourcesDir()
        imagepath = os.path.join(testresourcesdir, "multilayer_tif", "dotdotdot_lzw.tif")

        testimg_pil = Image.open(imagepath)
        testimg_arys = list()
        testimg_arys.append(array(testimg_pil))
        testimg_pil.seek(1)
        testimg_arys.append(array(testimg_pil))
        testimg_pil.seek(2)
        testimg_arys.append(array(testimg_pil))

        series = SeriesLoader(self.sc).fromMultipageTif(imagepath)
        assert_equals('float16', series._dtype)
        series_ary = series.pack()

        assert_equals((70, 75, 3), series.dims.count)
        assert_equals((70, 75, 3), series_ary.shape)
        assert_true(array_equal(testimg_arys[0], series_ary[:, :, 0]))
        assert_true(array_equal(testimg_arys[1], series_ary[:, :, 1]))
        assert_true(array_equal(testimg_arys[2], series_ary[:, :, 2]))

    def _run_fromFishTif(self, blocksize="150M"):
        imagepath = TestSeriesLoader._findSourceTreeDir("utils/data/fish/tif-stack")
        series = SeriesLoader(self.sc).fromMultipageTif(imagepath, blockSize=blocksize)
        assert_equals('float16', series._dtype)
        series_ary = series.pack()
        series_ary_xpose = series.pack(transpose=True)
        assert_equals('float16', str(series_ary.dtype))
        assert_equals((76, 87, 2), series.dims.count)
        assert_equals((20, 76, 87, 2), series_ary.shape)
        assert_equals((20, 2, 87, 76), series_ary_xpose.shape)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_fromFishTif(self):
        self._run_fromFishTif()

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_fromFishTifWithTinyBlocks(self):
        self._run_fromFishTif(blocksize=76*20)


class TestSeriesLoaderFromStacks(PySparkTestCaseWithOutputDir):
    def test_loadStacksAsSeries(self):
        rangeary = arange(64*128, dtype=dtype('int16'))
        rangeary.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeary.stack")
        rangeary.tofile(filepath)

        series = SeriesLoader(self.sc).fromStack(filepath, dims=(128, 64))
        series_ary = series.pack()

        assert_equals((128, 64), series.dims.count)
        assert_equals((128, 64), series_ary.shape)
        assert_true(array_equal(rangeary.T, series_ary))


class TestSeriesBinaryLoader(PySparkTestCaseWithOutputDir):

    def _run_tst_fromBinary(self, useConfJson=False):
        # run this as a single big test so as to avoid repeated setUp and tearDown of the spark context
        DATA = []
        # data will be a sequence of test data
        # all keys and all values in a test data item must be of the same length
        # keys get converted to ints regardless of raw input format
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int16'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3], [5, 6, 7]], [[11], [12]], 'int16', 'int16'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int32'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int32', 'int16'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11.0, 12.0, 13.0]], 'int16', 'float32'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11.0, 12.0, 13.0]], 'float32', 'float32'))
        DATA.append(SeriesBinaryTestData.fromArrays([[2, 3, 4]], [[11.0, 12.0, 13.0]], 'float32', 'float32'))

        for itemidx, item in enumerate(DATA):
            outsubdir = os.path.join(self.outputdir, 'input%d' % itemidx)
            os.mkdir(outsubdir)

            fname = os.path.join(outsubdir, 'inputfile%d.bin' % itemidx)
            with open(fname, 'wb') as f:
                item.writeToFile(f)

            loader = SeriesLoader(self.sc)
            if not useConfJson:
                series = loader.fromBinary(outsubdir, nkeys=item.nkeys, nvalues=item.nvals, keytype=str(item.keyDType),
                                           valuetype=str(item.valDType))
            else:
                # write configuration file
                conf = {'input': outsubdir,
                        'nkeys': item.nkeys, 'nvalues': item.nvals,
                        'valuetype': str(item.valDType), 'keytype': str(item.keyDType)}
                with open(os.path.join(outsubdir, "conf.json"), 'wb') as f:
                    json.dump(conf, f, indent=2)
                series = loader.fromBinary(outsubdir)

            seriesdata = series.rdd.collect()

            expecteddata = item.data
            assert_equals(len(expecteddata), len(seriesdata),
                          "Differing numbers of k/v pairs in item %d; expected %d, got %d" %
                          (itemidx, len(expecteddata), len(seriesdata)))

            for expected, actual in zip(expecteddata, seriesdata):
                expectedkeys = tuple(expected[0])
                expectedtype = smallestFloatType(item.valDType)
                expectedvals = array(expected[1], dtype=expectedtype)
                assert_equals(expectedkeys, actual[0],
                              "Key mismatch in item %d; expected %s, got %s" %
                              (itemidx, str(expectedkeys), str(actual[0])))
                assert_true(allclose(expectedvals, actual[1]),
                            "Value mismatch in item %d; expected %s, got %s" %
                            (itemidx, str(expectedvals), str(actual[1])))
                assert_equals(expectedtype, str(actual[1].dtype),
                              "Value type mismatch in item %d; expected %s, got %s" %
                              (itemidx, expectedtype, str(actual[1].dtype)))

    def test_fromBinary(self):
        self._run_tst_fromBinary()

    def test_fromBinaryWithConfFile(self):
        self._run_tst_fromBinary(True)


class TestSeriesBinaryWriteFromStack(PySparkTestCaseWithOutputDir):

    def _run_roundtrip_tst(self, testCount, arrays, blockSize):
        print "Running TestSeriesBinaryWriteFromStack roundtrip test #%d" % testCount
        insubdir = os.path.join(self.outputdir, 'input%d' % testCount)
        os.mkdir(insubdir)

        outsubdir = os.path.join(self.outputdir, 'output%d' % testCount)
        #os.mkdir(outsubdir)

        for aryCount, array in enumerate(arrays):
            # array.tofile always writes in column-major order...
            array.tofile(os.path.join(insubdir, "img%02d.stack" % aryCount))

        # ... but we will read and interpret these as though they are in row-major order
        dims = list(arrays[0].shape)
        dims.reverse()

        underTest = SeriesLoader(self.sc)

        underTest.saveFromStack(insubdir, outsubdir, dims, blockSize=blockSize, datatype=str(arrays[0].dtype))
        series = underTest.fromStack(insubdir, dims, datatype=str(arrays[0].dtype))

        roundtripped_series = underTest.fromBinary(outsubdir)
        roundtripped = roundtripped_series.collect()
        direct = series.collect()

        expecteddtype = str(smallestFloatType(arrays[0].dtype))
        assert_equals(expecteddtype, roundtripped_series.dtype)
        assert_equals(expecteddtype, series.dtype)
        assert_equals(expecteddtype, str(roundtripped[0][1].dtype))
        assert_equals(expecteddtype, str(direct[0][1].dtype))

        with open(os.path.join(outsubdir, "conf.json"), 'r') as fp:
            # check that binary series file data type *matches* input stack data type (not yet converted to float)
            # at least according to conf.json
            conf = json.load(fp)
            assert_equals(str(arrays[0].dtype), conf["valuetype"])

        for ((serieskeys, seriesvalues), (directkeys, directvalues)) in zip(roundtripped, direct):
            assert_equals(directkeys, serieskeys)
            assert_equals(directvalues, seriesvalues)

            for seriesidx, seriesval in enumerate(seriesvalues):
                #print "seriesidx: %d; serieskeys: %s; seriesval: %g" % (seriesidx, serieskeys, seriesval)
                # flip indices again for row vs col-major insanity
                arykeys = list(serieskeys)
                arykeys.reverse()
                msg = "Failure on test #%d, time point %d, indices %s" % (testCount, seriesidx, str(tuple(arykeys)))
                try:
                    assert_almost_equal(arrays[seriesidx][tuple(arykeys)], seriesval, places=4)
                except AssertionError, e:
                    raise AssertionError(msg, e)

    @staticmethod
    def generate_tst_images(narrays, dims, datatype):
        nimgvals = reduce(lambda x, y: x * y, dims)
        return [arange(i*nimgvals, (i+1)*nimgvals, dtype=datatype).reshape(dims) for i in xrange(narrays)]

    def _roundtrip_tst_driver(self, moreTests=False):
        # parameterized test fixture
        #arrays = [arange(6, dtype='int16').reshape((2, 3), order='F')]
        arrays = TestSeriesBinaryWriteFromStack.generate_tst_images(1, (2, 3), "int16")
        self._run_roundtrip_tst(0, arrays, 6*2)
        self._run_roundtrip_tst(1, arrays, 2*2)
        self._run_roundtrip_tst(2, arrays, 5*2)
        self._run_roundtrip_tst(3, arrays, 7*2)

        if moreTests:
            arrays = TestSeriesBinaryWriteFromStack.generate_tst_images(3, (5, 7, 5), "int16")
            self._run_roundtrip_tst(4, arrays, 3*5*2)

            arrays = TestSeriesBinaryWriteFromStack.generate_tst_images(3, (5, 7, 5), "int32")
            self._run_roundtrip_tst(5, arrays, 3*5*4)
            self._run_roundtrip_tst(6, arrays, 5*7*4)
            self._run_roundtrip_tst(7, arrays, 3*4)

            arrays = TestSeriesBinaryWriteFromStack.generate_tst_images(3, (2, 4, 6), "float32")
            self._run_roundtrip_tst(8, arrays, 5*4)

    def test_roundtrip(self):
        self._roundtrip_tst_driver(False)
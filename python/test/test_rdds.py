import shutil
import struct
import tempfile
import os
import logging
import unittest
from numpy import dtype, array, allclose, ndarray
from nose.tools import assert_equals, assert_true, assert_almost_equal
from thunder.rdds.series import SeriesLoader, Series
from thunder.rdds.images import ImagesLoader, Images
from test_utils import PySparkTestCase

_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    pass


class RDDsSparkTestCase(PySparkTestCase):
    def setUp(self):
        super(RDDsSparkTestCase, self).setUp()
        # suppress lots of DEBUG output from py4j
        logging.getLogger("py4j").setLevel(logging.WARNING)
        logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(RDDsSparkTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


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


class TestSeriesBinaryLoader(RDDsSparkTestCase):

    def test_fromBinary(self):
        # run this as a single big test so as to avoid repeated setUp and tearDown of the spark context
        DATA = []
        # data will be a sequence of test data
        # all keys and all values in a test data item must be of the same length
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int16'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3], [5, 6, 7]], [[11], [12]], 'int16', 'int16'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int32'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int32', 'int16'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11.0, 12.0, 13.0]], 'int16', 'float32'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1.0, 2.0, 3.0]], [[11.0, 12.0, 13.0]], 'float32', 'float32'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1.5, 2.5, 3.5]], [[11.0, 12.0, 13.0]], 'float32', 'float32'))

        for itemidx, item in enumerate(DATA):
            fname = os.path.join(self.outputdir, 'inputfile%d.bin' % itemidx)
            with open(fname, 'wb') as f:
                item.writeToFile(f)

            loader = SeriesLoader(self.sc, item.nkeys, item.nvals, keytype=str(item.keyDType), valuetype=str(item.valDType))
            series = loader.fromBinary(fname)
            seriesdata = series.rdd.collect()

            expecteddata = item.data
            assert_equals(len(expecteddata), len(seriesdata),
                          "Differing numbers of k/v pairs in item %d; expected %d, got %d" %
                          (itemidx, len(expecteddata), len(seriesdata)))

            for expected, actual in zip(expecteddata, seriesdata):
                expectedkeys = tuple(expected[0])
                expectedvals = array(expected[1], dtype=item.valDType)
                assert_equals(expectedkeys, actual[0],
                              "Key mismatch in item %d; expected %s, got %s" %
                              (itemidx, str(expectedkeys), str(actual[0])))
                assert_true(allclose(expectedvals, actual[1]),
                            "Value mismatch in item %d; expected %s, got %s" %
                            (itemidx, str(expectedvals), str(actual[1])))
                assert_equals(item.valDType, actual[1].dtype,
                              "Value type mismatch in item %d; expected %s, got %s" %
                              (itemidx, str(item.valDType), str(actual[1].dtype)))


class TestImagesFileLoaders(RDDsSparkTestCase):
    @staticmethod
    def _findTestResourcesDir(resourcesdirname="resources"):
        testdirpath = os.path.dirname(os.path.realpath(__file__))
        testresourcesdirpath = os.path.join(testdirpath, resourcesdirname)
        if not os.path.isdir(testresourcesdirpath):
            raise IOError("Test resources directory "+testresourcesdirpath+" not found")
        return testresourcesdirpath

    def setUp(self):
        super(TestImagesFileLoaders, self).setUp()
        self.testresourcesdir = self._findTestResourcesDir()

    def test_fromPng(self):
        imagepath = os.path.join(self.testresourcesdir, "singlelayer_png", "dot1.png")
        pngimage = ImagesLoader(self.sc).fromPng(imagepath, self.sc)
        firstpngimage = pngimage.first()
        assert_equals(0, firstpngimage[0], "Key error; expected first image key to be 0, was "+str(firstpngimage[0]))
        expectedshape = (70, 75, 4)  # 4 channel png; RGBalpha
        assert_true(isinstance(firstpngimage[1], ndarray),
                    "Value type error; expected first image value to be numpy ndarray, was " +
                    str(type(firstpngimage[1])))
        assert_equals(expectedshape, firstpngimage[1].shape)
        assert_almost_equal(0.97, firstpngimage[1][:, :, 0].flatten().max(), places=2)
        assert_almost_equal(0.03, firstpngimage[1][:, :, 0].flatten().min(), places=2)

    def test_fromTif(self):
        imagepath = os.path.join(self.testresourcesdir, "singlelayer_tif", "dot1_lzw.tif")
        tifimage = ImagesLoader(self.sc).fromTif(imagepath, self.sc)
        firsttifimage = tifimage.first()
        assert_equals(0, firsttifimage[0], "Key error; expected first image key to be 0, was "+str(firsttifimage[0]))
        expectedshape = (70, 75, 4)  # 4 channel tif; RGBalpha
        assert_true(isinstance(firsttifimage[1], ndarray),
                    "Value type error; expected first image value to be numpy ndarray, was " +
                    str(type(firsttifimage[1])))
        assert_equals(expectedshape, firsttifimage[1].shape)
        assert_equals(248, firsttifimage[1][:, :, 0].flatten().max())
        assert_equals(8, firsttifimage[1][:, :, 0].flatten().min())

    @staticmethod
    def _evaluateMultipleImages(tifimages, expectednum, expectedshape, expectedkeys, expectedsums):
        assert_equals(expectednum, len(tifimages), "Expected %s images, got %d" % (expectednum, len(tifimages)))
        for img, expectedkey, expectedsum in zip(tifimages, expectedkeys, expectedsums):
            assert_equals(expectedkey, img[0], "Expected key %s, got %s" % (str(expectedkey), str(img[0])))

            assert_true(isinstance(img[1], ndarray),
                        "Value type error; expected image value to be numpy ndarray, was " + str(type(img[1])))
            assert_equals(expectedshape, img[1].shape)
            assert_equals(expectedsum, img[1][:, :, 0].sum())

    def test_fromTifWithMultipleFiles(self):
        imagepath = os.path.join(self.testresourcesdir, "singlelayer_tif", "dot*_lzw.tif")
        tifimages = ImagesLoader(self.sc).fromTif(imagepath, self.sc).collect()

        expectednum = 3
        expectedshape = (70, 75, 4)  # 4 channel tif; RGBalpha
        expectedsums = [1282192, 1261328, 1241520]  # 3 images have increasing #s of black dots, so lower luminance overall
        expectedkeys = range(expectednum)
        self._evaluateMultipleImages(tifimages, expectednum, expectedshape, expectedkeys, expectedsums)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_fromMultipageTif(self):
        imagepath = os.path.join(self.testresourcesdir, "multilayer_tif", "dotdotdot_lzw.tif")
        tifimages = ImagesLoader(self.sc).fromMultipageTif(imagepath, self.sc).collect()

        expectednum = 1
        expectedshape = (70, 75, 4*3)  # 4 channel tifs of RGBalpha times 3 concatenated pages
        expectedsums = [1282192, 1261328, 1241520]  # 3 images have increasing #s of black dots, so lower luminance overall
        expectedkey = 0
        #self._evaluateMultipleImages(tifimages, expectednum, expectedshape, expectedkeys, expectedsums)

        assert_equals(expectednum, len(tifimages), "Expected %s images, got %d" % (expectednum, len(tifimages)))
        tifimage = tifimages[0]
        assert_equals(expectedkey, tifimage[0], "Expected key %s, got %s" % (str(expectedkey), str(tifimage[0])))
        assert_true(isinstance(tifimage[1], ndarray),
                    "Value type error; expected image value to be numpy ndarray, was " + str(type(tifimage[1])))
        assert_equals(expectedshape, tifimage[1].shape)
        for channelidx in xrange(0, expectedshape[2], 4):
            assert_equals(expectedsums[channelidx/4], tifimage[1][:, :, channelidx].flatten().sum())


class TestSeriesMethods(RDDsSparkTestCase):

    def test_between(self):
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd).between(0, 1)
        allclose(data.index, array([0, 1]))
        allclose(data.first()[1], array([4, 5]))

    def test_select(self):
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd, index=['label1', 'label2', 'label3', 'label4'])
        selection1 = data.select(['label1'])
        allclose(selection1.first()[1], array([4]))
        selection2 = data.select(['label1', 'label2'])
        allclose(selection2.first()[1], array([4, 5]))

    def test_detrend(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd).detrend('linear')
        # detrending linearly increasing data should yield all 0s
        allclose(data.first()[1], array([0, 0, 0, 0, 0]))

    def test_series_stats(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd)
        allclose(data.seriesMean().first()[1], 3.0)
        allclose(data.seriesSum().first()[1], 15.0)
        allclose(data.seriesStdev().first()[1], 1.4142135)
        allclose(data.seriesStat('mean').first()[1], 3.0)
        allclose(data.seriesStats().select('mean').first()[1], 3.0)
        allclose(data.seriesStats().select('count').first()[1], 5)

    def test_standardization(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd)
        allclose(data.center().first()[1], array([-2, -1, 0, 1, 2]))
        allclose(data.normalize().first()[1], array([-0.42105,  0.10526,  0.63157,  1.15789,  1.68421]))
        allclose(data.standardize().first()[1], array([0.70710,  1.41421,  2.12132,  2.82842,  3.53553]))
        allclose(data.zscore().first()[1], array([-1.41421, -0.70710,  0,  0.70710,  1.41421]))


if __name__ == "__main__":
    if not _have_image:
        print "NOTE: Skipping PIL/pillow tests as neither seem to be installed and functional"
    unittest.main()
    if not _have_image:
        print "NOTE: PIL/pillow tests were skipped as neither seem to be installed and functional"

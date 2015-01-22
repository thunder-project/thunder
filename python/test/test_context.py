import os
import unittest
from nose.tools import assert_equals, assert_true
from numpy import arange, array, array_equal, mod
from numpy import dtype as dtypeFunc

from test_utils import PySparkTestCaseWithOutputDir
from thunder import ThunderContext

_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    Image = None


class TestContextLoading(PySparkTestCaseWithOutputDir):
    def setUp(self):
        super(TestContextLoading, self).setUp()
        self.tsc = ThunderContext(self.sc)

    @staticmethod
    def _findTestResourcesDir(resourcesDirName="resources"):
        testDirPath = os.path.dirname(os.path.realpath(__file__))
        testResourcesDirPath = os.path.join(testDirPath, resourcesDirName)
        if not os.path.isdir(testResourcesDirPath):
            raise IOError("Test resources directory "+testResourcesDirPath+" not found")
        return testResourcesDirPath

    def __run_loadStacksAsSeries(self, shuffle):
        rangeAry = arange(64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary.stack")
        rangeAry.tofile(filePath)
        expectedAry = rangeAry.reshape((128, 64), order='F')

        rangeSeries = self.tsc.loadImagesAsSeries(filePath, dims=(128, 64), shuffle=shuffle)
        assert_equals('float32', rangeSeries._dtype)  # check before any potential first() calls update this val
        rangeSeriesAry = rangeSeries.pack()

        assert_equals((128, 64), rangeSeries.dims.count)
        assert_equals((128, 64), rangeSeriesAry.shape)
        assert_equals('float32', str(rangeSeriesAry.dtype))
        assert_true(array_equal(expectedAry, rangeSeriesAry))

    def test_loadStacksAsSeriesNoShuffle(self):
        self.__run_loadStacksAsSeries(False)

    def test_loadStacksAsSeriesWithShuffle(self):
        self.__run_loadStacksAsSeries(True)

    def __run_load3dStackAsSeries(self, shuffle):
        rangeAry = arange(32*64*4, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary.stack")
        rangeAry.tofile(filePath)
        expectedAry = rangeAry.reshape((32, 64, 4), order='F')

        rangeSeries = self.tsc.loadImagesAsSeries(filePath, dims=(32, 64, 4), shuffle=shuffle)
        assert_equals('float32', rangeSeries._dtype)
        rangeSeriesAry = rangeSeries.pack()

        assert_equals((32, 64, 4), rangeSeries.dims.count)
        assert_equals((32, 64, 4), rangeSeriesAry.shape)
        assert_equals('float32', str(rangeSeriesAry.dtype))
        assert_true(array_equal(expectedAry, rangeSeriesAry))

    def test_load3dStackAsSeriesNoShuffle(self):
        self.__run_load3dStackAsSeries(False)

    def test_load3dStackAsSeriesWithShuffle(self):
        self.__run_load3dStackAsSeries(True)

    def __run_loadMultipleStacksAsSeries(self, shuffle):
        rangeAry = arange(64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary01.stack")
        rangeAry.tofile(filePath)
        expectedAry = rangeAry.reshape((128, 64), order='F')
        rangeAry2 = arange(64*128, 2*64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary02.stack")
        rangeAry2.tofile(filePath)
        expectedAry2 = rangeAry2.reshape((128, 64), order='F')

        rangeSeries = self.tsc.loadImagesAsSeries(self.outputdir, dims=(128, 64), shuffle=shuffle)
        assert_equals('float32', rangeSeries._dtype)

        rangeSeriesAry = rangeSeries.pack()
        rangeSeriesAry_xpose = rangeSeries.pack(transpose=True)

        assert_equals((128, 64), rangeSeries.dims.count)
        assert_equals((2, 128, 64), rangeSeriesAry.shape)
        assert_equals((2, 64, 128), rangeSeriesAry_xpose.shape)
        assert_equals('float32', str(rangeSeriesAry.dtype))
        assert_true(array_equal(expectedAry, rangeSeriesAry[0]))
        assert_true(array_equal(expectedAry2, rangeSeriesAry[1]))
        assert_true(array_equal(expectedAry.T, rangeSeriesAry_xpose[0]))
        assert_true(array_equal(expectedAry2.T, rangeSeriesAry_xpose[1]))

    def test_loadMultipleStacksAsSeriesNoShuffle(self):
        self.__run_loadMultipleStacksAsSeries(False)

    def test_loadMultipleStacksAsSeriesWithShuffle(self):
        self.__run_loadMultipleStacksAsSeries(True)

    def test_loadMultipleMultipointStacksAsSeries(self):
        rangeAry = arange(64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary01.stack")
        rangeAry.tofile(filePath)
        expectedAry = rangeAry.reshape((32, 32, 8), order='F')
        rangeAry2 = arange(64*128, 2*64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary02.stack")
        rangeAry2.tofile(filePath)
        expectedAry2 = rangeAry2.reshape((32, 32, 8), order='F')

        rangeSeries = self.tsc.loadImagesAsSeries(self.outputdir, dims=(32, 32, 8), nplanes=2, shuffle=True)
        assert_equals('float32', rangeSeries._dtype)

        rangeSeriesAry = rangeSeries.pack()

        assert_equals((32, 32, 2), rangeSeries.dims.count)
        assert_equals((8, 32, 32, 2), rangeSeriesAry.shape)
        assert_equals('float32', str(rangeSeriesAry.dtype))
        assert_true(array_equal(expectedAry[:, :, :2], rangeSeriesAry[0]))
        assert_true(array_equal(expectedAry[:, :, 2:4], rangeSeriesAry[1]))
        assert_true(array_equal(expectedAry[:, :, 4:6], rangeSeriesAry[2]))
        assert_true(array_equal(expectedAry[:, :, 6:], rangeSeriesAry[3]))
        assert_true(array_equal(expectedAry2[:, :, :2], rangeSeriesAry[4]))
        assert_true(array_equal(expectedAry2[:, :, 2:4], rangeSeriesAry[5]))
        assert_true(array_equal(expectedAry2[:, :, 4:6], rangeSeriesAry[6]))
        assert_true(array_equal(expectedAry2[:, :, 6:], rangeSeriesAry[7]))

    def __run_loadTifAsSeries(self, shuffle):
        tmpAry = arange(60*120, dtype=dtypeFunc('uint16'))
        rangeAry = mod(tmpAry, 255).astype('uint8').reshape((60, 120))
        pilImg = Image.fromarray(rangeAry)
        filePath = os.path.join(self.outputdir, "rangetif01.tif")
        pilImg.save(filePath)
        del pilImg, tmpAry

        rangeSeries = self.tsc.loadImagesAsSeries(self.outputdir, inputFormat="tif-stack", shuffle=shuffle)
        assert_equals('float16', rangeSeries._dtype)  # check before any potential first() calls update this val
        rangeSeriesAry = rangeSeries.pack()

        assert_equals((60, 120, 1), rangeSeries.dims.count)
        assert_equals((60, 120), rangeSeriesAry.shape)
        assert_equals('float16', str(rangeSeriesAry.dtype))
        assert_true(array_equal(rangeAry, rangeSeriesAry))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTifAsSeriesNoShuffle(self):
        self.__run_loadTifAsSeries(False)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTifAsSeriesWithShuffle(self):
        self.__run_loadTifAsSeries(True)

    def __run_loadTestTifAsSeries(self, shuffle):
        testResourcesDir = TestContextLoading._findTestResourcesDir()
        imagePath = os.path.join(testResourcesDir, "multilayer_tif", "dotdotdot_lzw.tif")

        testimg_pil = Image.open(imagePath)
        testimg_arys = list()
        testimg_arys.append(array(testimg_pil))  # original shape 70, 75
        testimg_pil.seek(1)
        testimg_arys.append(array(testimg_pil))
        testimg_pil.seek(2)
        testimg_arys.append(array(testimg_pil))

        rangeSeries = self.tsc.loadImagesAsSeries(imagePath, inputFormat="tif-stack", shuffle=shuffle)
        assert_true(rangeSeries._dtype.startswith("float"))
        rangeSeriesAry = rangeSeries.pack()
        rangeSeriesAry_xpose = rangeSeries.pack(transpose=True)

        assert_equals((70, 75, 3), rangeSeries.dims.count)
        assert_equals((70, 75, 3), rangeSeriesAry.shape)
        assert_equals((3, 75, 70), rangeSeriesAry_xpose.shape)
        assert_true(rangeSeriesAry.dtype.kind == "f")
        assert_true(array_equal(testimg_arys[0], rangeSeriesAry[:, :, 0]))
        assert_true(array_equal(testimg_arys[1], rangeSeriesAry[:, :, 1]))
        assert_true(array_equal(testimg_arys[2], rangeSeriesAry[:, :, 2]))
        assert_true(array_equal(testimg_arys[0].T, rangeSeriesAry_xpose[0]))
        assert_true(array_equal(testimg_arys[1].T, rangeSeriesAry_xpose[1]))
        assert_true(array_equal(testimg_arys[2].T, rangeSeriesAry_xpose[2]))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTestTifAsSeriesNoShuffle(self):
        self.__run_loadTestTifAsSeries(False)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTestTifAsSeriesWithShuffle(self):
        self.__run_loadTestTifAsSeries(True)

    def __run_loadMultipleTifsAsSeries(self, shuffle):
        tmpAry = arange(60*120, dtype=dtypeFunc('uint16'))
        rangeAry = mod(tmpAry, 255).astype('uint8').reshape((60, 120))
        pilImg = Image.fromarray(rangeAry)
        filePath = os.path.join(self.outputdir, "rangetif01.tif")
        pilImg.save(filePath)

        tmpAry = arange(60*120, 2*60*120, dtype=dtypeFunc('uint16'))
        rangeAry2 = mod(tmpAry, 255).astype('uint8').reshape((60, 120))
        pilImg = Image.fromarray(rangeAry2)
        filePath = os.path.join(self.outputdir, "rangetif02.tif")
        pilImg.save(filePath)

        del pilImg, tmpAry

        rangeSeries = self.tsc.loadImagesAsSeries(self.outputdir, inputFormat="tif-stack", shuffle=shuffle)
        assert_equals('float16', rangeSeries._dtype)
        rangeSeriesAry = rangeSeries.pack()
        rangeSeriesAry_xpose = rangeSeries.pack(transpose=True)

        assert_equals((60, 120, 1), rangeSeries.dims.count)
        assert_equals((2, 60, 120), rangeSeriesAry.shape)
        assert_equals((2, 120, 60), rangeSeriesAry_xpose.shape)
        assert_equals('float16', str(rangeSeriesAry.dtype))
        assert_true(array_equal(rangeAry, rangeSeriesAry[0]))
        assert_true(array_equal(rangeAry2, rangeSeriesAry[1]))
        assert_true(array_equal(rangeAry.T, rangeSeriesAry_xpose[0]))
        assert_true(array_equal(rangeAry2.T, rangeSeriesAry_xpose[1]))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadMultipleTifsAsSeriesNoShuffle(self):
        self.__run_loadMultipleTifsAsSeries(False)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadMultipleTifsAsSeriesWithShuffle(self):
        self.__run_loadMultipleTifsAsSeries(True)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadMultipleMultipointTifsAsSeries(self):
        testResourcesDir = TestContextLoading._findTestResourcesDir()
        imagesPath = os.path.join(testResourcesDir, "multilayer_tif", "dotdotdot_lzw*.tif")

        # load only one file, second is a copy of this one
        testimg_pil = Image.open(os.path.join(testResourcesDir, "multilayer_tif", "dotdotdot_lzw.tif"))
        testimg_arys = [array(testimg_pil)]
        for idx in xrange(1, 3):
            testimg_pil.seek(idx)
            testimg_arys.append(array(testimg_pil))

        rangeSeries = self.tsc.loadImagesAsSeries(imagesPath, inputFormat="tif-stack", shuffle=True, nplanes=1)
        assert_equals((70, 75), rangeSeries.dims.count)

        rangeSeriesAry = rangeSeries.pack()
        assert_equals((6, 70, 75), rangeSeriesAry.shape)
        for idx in xrange(6):
            assert_true(array_equal(testimg_arys[idx % 3], rangeSeriesAry[idx]))

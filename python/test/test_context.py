import numpy as np
import os
import unittest
from nose.tools import assert_equals, assert_true

from test_utils import PySparkTestCaseWithOutputDir
from thunder import ThunderContext
from thunder.utils.common import pil_to_array

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
    def _findTestResourcesDir(resourcesdirname="resources"):
        testdirpath = os.path.dirname(os.path.realpath(__file__))
        testresourcesdirpath = os.path.join(testdirpath, resourcesdirname)
        if not os.path.isdir(testresourcesdirpath):
            raise IOError("Test resources directory "+testresourcesdirpath+" not found")
        return testresourcesdirpath

    def __run_loadStacksAsSeries(self, shuffle):
        rangeary = np.arange(64*128, dtype=np.dtype('int16'))
        filepath = os.path.join(self.outputdir, "rangeary.stack")
        rangeary.tofile(filepath)
        expectedary = rangeary.reshape((128, 64), order='F')

        range_series = self.tsc.loadImagesAsSeries(filepath, dims=(128, 64), shuffle=shuffle)
        range_series_ary = range_series.pack()

        assert_equals((128, 64), range_series.dims.count)
        assert_equals((128, 64), range_series_ary.shape)
        assert_true(np.array_equal(expectedary, range_series_ary))

    def test_loadStacksAsSeriesNoShuffle(self):
        self.__run_loadStacksAsSeries(False)

    def test_loadStacksAsSeriesWithShuffle(self):
        self.__run_loadStacksAsSeries(True)

    def __run_load3dStackAsSeries(self, shuffle):
        rangeary = np.arange(32*64*4, dtype=np.dtype('int16'))
        filepath = os.path.join(self.outputdir, "rangeary.stack")
        rangeary.tofile(filepath)
        expectedary = rangeary.reshape((32, 64, 4), order='F')

        range_series_noshuffle = self.tsc.loadImagesAsSeries(filepath, dims=(32, 64, 4), shuffle=shuffle)
        range_series_noshuffle_ary = range_series_noshuffle.pack()

        assert_equals((32, 64, 4), range_series_noshuffle.dims.count)
        assert_equals((32, 64, 4), range_series_noshuffle_ary.shape)
        assert_true(np.array_equal(expectedary, range_series_noshuffle_ary))

    def test_load3dStackAsSeriesNoShuffle(self):
        self.__run_load3dStackAsSeries(False)

    def test_load3dStackAsSeriesWithShuffle(self):
        self.__run_load3dStackAsSeries(True)

    def __run_loadMultipleStacksAsSeries(self, shuffle):
        rangeary = np.arange(64*128, dtype=np.dtype('int16'))
        filepath = os.path.join(self.outputdir, "rangeary01.stack")
        rangeary.tofile(filepath)
        expectedary = rangeary.reshape((128, 64), order='F')
        rangeary2 = np.arange(64*128, 2*64*128, dtype=np.dtype('int16'))
        filepath = os.path.join(self.outputdir, "rangeary02.stack")
        rangeary2.tofile(filepath)
        expectedary2 = rangeary2.reshape((128, 64), order='F')

        range_series = self.tsc.loadImagesAsSeries(self.outputdir, dims=(128, 64), shuffle=shuffle)
        range_series_ary = range_series.pack()
        range_series_ary_xpose = range_series.pack(transpose=True)

        assert_equals((128, 64), range_series.dims.count)
        assert_equals((2, 128, 64), range_series_ary.shape)
        assert_equals((2, 64, 128), range_series_ary_xpose.shape)
        assert_true(np.array_equal(expectedary, range_series_ary[0]))
        assert_true(np.array_equal(expectedary2, range_series_ary[1]))
        assert_true(np.array_equal(expectedary.T, range_series_ary_xpose[0]))
        assert_true(np.array_equal(expectedary2.T, range_series_ary_xpose[1]))

    def test_loadMultipleStacksAsSeriesNoShuffle(self):
        self.__run_loadMultipleStacksAsSeries(False)

    def test_loadMultipleStacksAsSeriesWithShuffle(self):
        self.__run_loadMultipleStacksAsSeries(True)

    def __run_loadTifAsSeries(self, shuffle):
        tmpary = np.arange(60*120, dtype=np.dtype('uint16'))
        rangeary = np.mod(tmpary, 255).astype('uint8').reshape((60, 120))
        pilimg = Image.fromarray(rangeary)
        filepath = os.path.join(self.outputdir, "rangetif01.tif")
        pilimg.save(filepath)
        del pilimg, tmpary

        range_series = self.tsc.loadImagesAsSeries(self.outputdir, inputformat="tif-stack", shuffle=shuffle)
        range_series_ary = range_series.pack()

        assert_equals((60, 120, 1), range_series.dims.count)
        assert_equals((60, 120), range_series_ary.shape)
        assert_true(np.array_equal(rangeary, range_series_ary))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTifAsSeriesNoShuffle(self):
        self.__run_loadTifAsSeries(False)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTifAsSeriesWithShuffle(self):
        self.__run_loadTifAsSeries(True)

    def __run_loadTestTifAsSeries(self, shuffle):
        testresourcesdir = TestContextLoading._findTestResourcesDir()
        imagepath = os.path.join(testresourcesdir, "multilayer_tif", "dotdotdot_lzw.tif")

        testimg_pil = Image.open(imagepath)
        testimg_arys = list()
        testimg_arys.append(pil_to_array(testimg_pil))  # original shape 70, 75
        testimg_pil.seek(1)
        testimg_arys.append(pil_to_array(testimg_pil))
        testimg_pil.seek(2)
        testimg_arys.append(pil_to_array(testimg_pil))

        range_series_noshuffle = self.tsc.loadImagesAsSeries(imagepath, inputformat="tif-stack", shuffle=shuffle)
        range_series_noshuffle_ary = range_series_noshuffle.pack()
        range_series_noshuffle_ary_xpose = range_series_noshuffle.pack(transpose=True)

        assert_equals((70, 75, 3), range_series_noshuffle.dims.count)
        assert_equals((70, 75, 3), range_series_noshuffle_ary.shape)
        assert_equals((3, 75, 70), range_series_noshuffle_ary_xpose.shape)
        assert_true(np.array_equal(testimg_arys[0], range_series_noshuffle_ary[:, :, 0]))
        assert_true(np.array_equal(testimg_arys[1], range_series_noshuffle_ary[:, :, 1]))
        assert_true(np.array_equal(testimg_arys[2], range_series_noshuffle_ary[:, :, 2]))
        assert_true(np.array_equal(testimg_arys[0].T, range_series_noshuffle_ary_xpose[0]))
        assert_true(np.array_equal(testimg_arys[1].T, range_series_noshuffle_ary_xpose[1]))
        assert_true(np.array_equal(testimg_arys[2].T, range_series_noshuffle_ary_xpose[2]))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTestTifAsSeriesNoShuffle(self):
        self.__run_loadTestTifAsSeries(False)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTestTifAsSeriesWithShuffle(self):
        self.__run_loadTestTifAsSeries(True)

    def __run_loadMultipleTifsAsSeries(self, shuffle):
        tmpary = np.arange(60*120, dtype=np.dtype('uint16'))
        rangeary = np.mod(tmpary, 255).astype('uint8').reshape((60, 120))
        pilimg = Image.fromarray(rangeary)
        filepath = os.path.join(self.outputdir, "rangetif01.tif")
        pilimg.save(filepath)

        tmpary = np.arange(60*120, 2*60*120, dtype=np.dtype('uint16'))
        rangeary2 = np.mod(tmpary, 255).astype('uint8').reshape((60, 120))
        pilimg = Image.fromarray(rangeary2)
        filepath = os.path.join(self.outputdir, "rangetif02.tif")
        pilimg.save(filepath)

        del pilimg, tmpary

        range_series = self.tsc.loadImagesAsSeries(self.outputdir, inputformat="tif-stack", shuffle=shuffle)
        range_series_ary = range_series.pack()
        range_series_ary_xpose = range_series.pack(transpose=True)

        assert_equals((60, 120, 1), range_series.dims.count)
        assert_equals((2, 60, 120), range_series_ary.shape)
        assert_equals((2, 120, 60), range_series_ary_xpose.shape)
        assert_true(np.array_equal(rangeary, range_series_ary[0]))
        assert_true(np.array_equal(rangeary2, range_series_ary[1]))
        assert_true(np.array_equal(rangeary.T, range_series_ary_xpose[0]))
        assert_true(np.array_equal(rangeary2.T, range_series_ary_xpose[1]))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadMultipleTifsAsSeriesNoShuffle(self):
        self.__run_loadMultipleTifsAsSeries(False)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadMultipleTifsAsSeriesWithShuffle(self):
        self.__run_loadMultipleTifsAsSeries(True)
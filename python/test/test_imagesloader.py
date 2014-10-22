from numpy import arange, array_equal, dtype, ndarray
import os
import unittest
from nose.tools import assert_equals, assert_true, assert_almost_equal

from thunder.rdds.fileio.imagesloader import ImagesLoader
from test_images import _have_image
from test_utils import PySparkTestCase


class TestImagesFileLoaders(PySparkTestCase):
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

    def test_fromArrays(self):
        ary = arange(8, dtype=dtype('int16')).reshape((2, 4))

        image = ImagesLoader(self.sc).fromArrays(ary)

        collectedimage = image.collect()
        assert_equals(1, len(collectedimage))
        assert_equals(0, collectedimage[0][0])  # check key
        assert_true(array_equal(ary, collectedimage[0][1]))  # check value

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
        expectedshape = (70, 75, 3)  # 3 concatenated pages, each with single luminance channel
        # 3 images have increasing #s of black dots, so lower luminance overall
        expectedsums = [1140006, 1119161, 1098917]
        expectedkey = 0
        #self._evaluateMultipleImages(tifimages, expectednum, expectedshape, expectedkeys, expectedsums)

        assert_equals(expectednum, len(tifimages), "Expected %s images, got %d" % (expectednum, len(tifimages)))
        tifimage = tifimages[0]
        assert_equals(expectedkey, tifimage[0], "Expected key %s, got %s" % (str(expectedkey), str(tifimage[0])))
        assert_true(isinstance(tifimage[1], ndarray),
                    "Value type error; expected image value to be numpy ndarray, was " + str(type(tifimage[1])))
        assert_equals('uint8', str(tifimage[1].dtype))
        assert_equals(expectedshape, tifimage[1].shape)
        for channelidx in xrange(0, expectedshape[2]):
            assert_equals(expectedsums[channelidx], tifimage[1][:, :, channelidx].flatten().sum())
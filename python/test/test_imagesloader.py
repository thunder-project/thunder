from numpy import arange, array_equal, ndarray
from numpy import dtype as dtypeFunc
import os
import unittest
from nose.tools import assert_equals, assert_true, assert_almost_equal

from thunder.rdds.fileio.imagesloader import ImagesLoader
from test_utils import PySparkTestCase, PySparkTestCaseWithOutputDir

_haveImage = False
try:
    from PIL import Image
    _haveImage = True
except ImportError:
    # PIL not available; skip tests that require it
    Image = None


class TestImagesFileLoaders(PySparkTestCase):
    @staticmethod
    def _findTestResourcesDir(resourcesdirname="resources"):
        testDirPath = os.path.dirname(os.path.realpath(__file__))
        testResourcesDirPath = os.path.join(testDirPath, resourcesdirname)
        if not os.path.isdir(testResourcesDirPath):
            raise IOError("Test resources directory "+testResourcesDirPath+" not found")
        return testResourcesDirPath

    def setUp(self):
        super(TestImagesFileLoaders, self).setUp()
        self.testResourcesDir = self._findTestResourcesDir()

    def test_fromArrays(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))

        image = ImagesLoader(self.sc).fromArrays(ary)

        collectedImage = image.collect()
        assert_equals(1, len(collectedImage))
        assert_equals(ary.shape, image.dims.count)
        assert_equals(0, collectedImage[0][0])  # check key
        assert_true(array_equal(ary, collectedImage[0][1]))  # check value

    def test_fromOCP(self):
      from urllib2 import urlopen, Request, URLError
      try:
        request = Request ("http://ocp.me/ocp/ca/freeman14/info/")
        response = urlopen(request)
        imagePath = "freeman14"
        ocpImage = ImagesLoader(self.sc).fromOCP(imagePath,startIdx=0,stopIdx=1,minBound=(0,0,0),maxBound=(128,128,16),resolution=0)
        assert_equals(ocpImage[1].shape,(128,128,16))
      except URLError, e:
        print "fromOCP is unavaliable"


    def test_fromPng(self):
        imagePath = os.path.join(self.testResourcesDir, "singlelayer_png", "dot1_grey.png")
        pngImage = ImagesLoader(self.sc).fromPng(imagePath)
        firstPngImage = pngImage.first()
        assert_equals(0, firstPngImage[0], "Key error; expected first image key to be 0, was "+str(firstPngImage[0]))
        expectedShape = (70, 75)
        assert_true(isinstance(firstPngImage[1], ndarray),
                    "Value type error; expected first image value to be numpy ndarray, was " +
                    str(type(firstPngImage[1])))
        assert_equals(expectedShape, firstPngImage[1].shape)
        assert_equals(expectedShape, pngImage.dims.count)
        assert_almost_equal(0.937, firstPngImage[1].ravel().max(), places=2)  # integer val 239
        assert_almost_equal(0.00392, firstPngImage[1].ravel().min(), places=2)  # integer val 1

    def test_fromTif(self):
        imagePath = os.path.join(self.testResourcesDir, "singlelayer_tif", "dot1_grey_lzw.tif")
        tiffImage = ImagesLoader(self.sc).fromTif(imagePath, self.sc)
        firstTiffImage = tiffImage.first()
        assert_equals(0, firstTiffImage[0], "Key error; expected first image key to be 0, was "+str(firstTiffImage[0]))
        expectedShape = (70, 75)
        assert_true(isinstance(firstTiffImage[1], ndarray),
                    "Value type error; expected first image value to be numpy ndarray, was " +
                    str(type(firstTiffImage[1])))
        assert_equals(expectedShape, firstTiffImage[1].shape)
        assert_equals(expectedShape, tiffImage.dims.count)
        assert_equals(239, firstTiffImage[1].ravel().max())
        assert_equals(1, firstTiffImage[1].ravel().min())

    @staticmethod
    def _evaluateMultipleImages(tiffImages, expectedNum, expectedShape, expectedKeys, expectedSums):
        assert_equals(expectedNum, len(tiffImages), "Expected %s images, got %d" % (expectedNum, len(tiffImages)))
        for img, expectedKey, expectedSum in zip(tiffImages, expectedKeys, expectedSums):
            assert_equals(expectedKey, img[0], "Expected key %s, got %s" % (str(expectedKey), str(img[0])))

            assert_true(isinstance(img[1], ndarray),
                        "Value type error; expected image value to be numpy ndarray, was " + str(type(img[1])))
            assert_equals(expectedShape, img[1].shape)
            assert_equals(expectedSum, img[1].sum())

    def test_fromTifWithMultipleFiles(self):
        imagePath = os.path.join(self.testResourcesDir, "singlelayer_tif", "dot*_grey_lzw.tif")
        tiffImages = ImagesLoader(self.sc).fromTif(imagePath, self.sc).collect()

        expectedNum = 3
        expectedShape = (70, 75)
        expectedSums = [1233881, 1212169, 1191300]
        # 3 images have increasing #s of black dots, so lower luminance overall
        expectedKeys = range(expectedNum)
        self._evaluateMultipleImages(tiffImages, expectedNum, expectedShape, expectedKeys, expectedSums)

    def _run_tst_multitif(self, filename, expectedDtype):
        imagePath = os.path.join(self.testResourcesDir, "multilayer_tif", filename)
        tiffImages = ImagesLoader(self.sc).fromTif(imagePath, self.sc).collect()

        expectedNum = 1
        expectedShape = (70, 75, 3)  # 3 concatenated pages, each with single luminance channel
        # 3 images have increasing #s of black dots, so lower luminance overall
        expectedSums = [1140006, 1119161, 1098917]
        expectedKey = 0

        assert_equals(expectedNum, len(tiffImages), "Expected %s images, got %d" % (expectedNum, len(tiffImages)))
        tiffImage = tiffImages[0]
        assert_equals(expectedKey, tiffImage[0], "Expected key %s, got %s" % (str(expectedKey), str(tiffImage[0])))
        assert_true(isinstance(tiffImage[1], ndarray),
                    "Value type error; expected image value to be numpy ndarray, was " + str(type(tiffImage[1])))
        assert_equals(expectedDtype, str(tiffImage[1].dtype))
        assert_equals(expectedShape, tiffImage[1].shape)
        for channelidx in xrange(0, expectedShape[2]):
            assert_equals(expectedSums[channelidx], tiffImage[1][:, :, channelidx].flatten().sum())

    @unittest.skipIf(not _haveImage, "PIL/pillow not installed or not functional")
    def test_fromMultipageTif(self):
        self._run_tst_multitif("dotdotdot_lzw.tif", "uint8")

    @unittest.skipIf(not _haveImage, "PIL/pillow not installed or not functional")
    def test_fromFloatingpointTif(self):
        self._run_tst_multitif("dotdotdot_float32.tif", "float32")


class TestImagesLoaderUsingOutputDir(PySparkTestCaseWithOutputDir):
    def test_fromStack(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))
        filename = os.path.join(self.outputdir, "test.stack")
        ary.tofile(filename)

        image = ImagesLoader(self.sc).fromStack(filename, dims=(4, 2))

        collectedImage = image.collect()
        assert_equals(1, len(collectedImage))
        assert_equals(0, collectedImage[0][0])  # check key
        # assert that image shape *matches* that in image dimensions:
        assert_equals(image.dims.count, collectedImage[0][1].shape)
        assert_true(array_equal(ary.T, collectedImage[0][1]))  # check value

    def test_fromStacks(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))
        ary2 = arange(8, 16, dtype=dtypeFunc('int16')).reshape((2, 4))
        filename = os.path.join(self.outputdir, "test01.stack")
        ary.tofile(filename)
        filename = os.path.join(self.outputdir, "test02.stack")
        ary2.tofile(filename)

        image = ImagesLoader(self.sc).fromStack(self.outputdir, dims=(4, 2))

        collectedImage = image.collect()
        assert_equals(2, len(collectedImage))
        assert_equals(0, collectedImage[0][0])  # check key
        assert_equals(image.dims.count, collectedImage[0][1].shape)
        assert_true(array_equal(ary.T, collectedImage[0][1]))  # check value
        assert_equals(1, collectedImage[1][0])  # check image 2
        assert_true(array_equal(ary2.T, collectedImage[1][1]))

import glob
import struct
import os
from numpy import allclose, arange, array, array_equal, prod, squeeze, zeros, size
from numpy import dtype as dtypeFunc
import itertools
from nose.tools import assert_equals, assert_raises, assert_true
import unittest

from thunder.rdds.series import Series
from thunder.rdds.images import Images
from thunder.rdds.timeseries import TimeSeries
from thunder.rdds.fileio.imagesloader import ImagesLoader
from thunder.rdds.fileio.seriesloader import SeriesLoader
from thunder.rdds.imgblocks.strategy import PaddedBlockingStrategy, SimpleBlockingStrategy
from test_utils import PySparkTestCase, PySparkTestCaseWithOutputDir

_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    Image = None


def _generateTestArrays(narys, dtype_='int16'):
    sh = 4, 3, 3
    sz = reduce(lambda x, y: x * y, sh, 1)
    arys = [arange(i, i+sz, dtype=dtypeFunc(dtype_)).reshape(sh) for i in xrange(0, sz * narys, sz)]
    return arys, sh, sz


def findSourceTreeDir(dirname="utils/data"):
    testdirpath = os.path.dirname(os.path.realpath(__file__))
    testresourcesdirpath = os.path.join(testdirpath, "..", "thunder", dirname)
    if not os.path.isdir(testresourcesdirpath):
        raise IOError("Directory "+testresourcesdirpath+" not found")
    return testresourcesdirpath


class TestImages(PySparkTestCase):

    def evaluateSeries(self, arys, series, sz):
        assert_equals(sz, len(series))
        for seriesKey, seriesVal in series:
            expectedVal = array([ary[seriesKey] for ary in arys], dtype='int16')
            assert_true(array_equal(expectedVal, seriesVal))

    def test_castToFloat(self):
        arys, shape, size = _generateTestArrays(2, 'uint8')
        imageData = ImagesLoader(self.sc).fromArrays(arys)
        catData = imageData.astype("smallfloat")

        assert_equals('float16', str(catData.dtype))
        assert_equals('float16', str(catData.first()[1].dtype))

    def test_toSeries(self):
        # create 3 arrays of 4x3x3 images (C-order), containing sequential integers
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)

        imageData = ImagesLoader(self.sc).fromArrays(arys)
        series = imageData.toBlocks((4, 1, 1), units="s").toSeries().collect()

        self.evaluateSeries(arys, series, sz)

    def test_toSeriesDirect(self):
        # create 3 arrays of 4x3x3 images (C-order), containing sequential integers
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)

        imageData = ImagesLoader(self.sc).fromArrays(arys)
        series = imageData.toSeries()

        assert(isinstance(series, Series))

    def test_toTimeSeries(self):
        # create 3 arrays of 4x3x3 images (C-order), containing sequential integers
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)

        imageData = ImagesLoader(self.sc).fromArrays(arys)
        series = imageData.toTimeSeries()

        assert(isinstance(series, TimeSeries))

    def test_toSeriesWithPack(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))

        image = ImagesLoader(self.sc).fromArrays(ary)
        series = image.toBlocks("150M").toSeries()

        seriesVals = series.collect()
        seriesAry = series.pack()
        seriesAry_xpose = series.pack(transpose=True)

        # check ordering of keys
        assert_equals((0, 0), seriesVals[0][0])  # first key
        assert_equals((1, 0), seriesVals[1][0])  # second key
        assert_equals((0, 1), seriesVals[2][0])
        assert_equals((1, 1), seriesVals[3][0])
        assert_equals((0, 2), seriesVals[4][0])
        assert_equals((1, 2), seriesVals[5][0])
        assert_equals((0, 3), seriesVals[6][0])
        assert_equals((1, 3), seriesVals[7][0])

        # check dimensions tuple matches numpy shape
        assert_equals(image.dims.count, series.dims.count)
        assert_equals(ary.shape, series.dims.count)

        # check that values are in Fortran-convention order
        collectedVals = array([kv[1] for kv in seriesVals], dtype=dtypeFunc('int16')).ravel()
        assert_true(array_equal(ary.ravel(order='F'), collectedVals))

        # check that packing returns original array
        assert_true(array_equal(ary, seriesAry))
        assert_true(array_equal(ary.T, seriesAry_xpose))

    def test_threeDArrayToSeriesWithPack(self):
        ary = arange(24, dtype=dtypeFunc('int16')).reshape((3, 4, 2))

        image = ImagesLoader(self.sc).fromArrays(ary)
        series = image.toBlocks("150M").toSeries()

        seriesVals = series.collect()
        seriesAry = series.pack()
        seriesAry_xpose = series.pack(transpose=True)

        # check ordering of keys
        assert_equals((0, 0, 0), seriesVals[0][0])  # first key
        assert_equals((1, 0, 0), seriesVals[1][0])  # second key
        assert_equals((2, 0, 0), seriesVals[2][0])
        assert_equals((0, 1, 0), seriesVals[3][0])
        assert_equals((1, 1, 0), seriesVals[4][0])
        assert_equals((2, 1, 0), seriesVals[5][0])
        assert_equals((0, 2, 0), seriesVals[6][0])
        assert_equals((1, 2, 0), seriesVals[7][0])
        assert_equals((2, 2, 0), seriesVals[8][0])
        assert_equals((0, 3, 0), seriesVals[9][0])
        assert_equals((1, 3, 0), seriesVals[10][0])
        assert_equals((2, 3, 0), seriesVals[11][0])
        assert_equals((0, 0, 1), seriesVals[12][0])
        assert_equals((1, 0, 1), seriesVals[13][0])
        assert_equals((2, 0, 1), seriesVals[14][0])
        assert_equals((0, 1, 1), seriesVals[15][0])
        assert_equals((1, 1, 1), seriesVals[16][0])
        assert_equals((2, 1, 1), seriesVals[17][0])
        assert_equals((0, 2, 1), seriesVals[18][0])
        assert_equals((1, 2, 1), seriesVals[19][0])
        assert_equals((2, 2, 1), seriesVals[20][0])
        assert_equals((0, 3, 1), seriesVals[21][0])
        assert_equals((1, 3, 1), seriesVals[22][0])
        assert_equals((2, 3, 1), seriesVals[23][0])

        # check dimensions tuple matches numpy shape
        assert_equals(ary.shape, series.dims.count)

        # check that values are in Fortran-convention order
        collectedVals = array([kv[1] for kv in seriesVals], dtype=dtypeFunc('int16')).ravel()
        assert_true(array_equal(ary.ravel(order='F'), collectedVals))

        # check that packing returns transpose of original array
        assert_true(array_equal(ary, seriesAry))
        assert_true(array_equal(ary.T, seriesAry_xpose))

    def _run_tst_toSeriesWithSplitsAndPack(self, strategy):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((4, 2))
        image = ImagesLoader(self.sc).fromArrays(ary)
        series = image.toBlocks(strategy).toSeries()

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

        # check dimensions tuple matches numpy shape
        assert_equals(ary.shape, series.dims.count)

        # check that values are in Fortran-convention order
        collectedVals = array([kv[1] for kv in seriesVals], dtype=dtypeFunc('int16')).ravel()
        assert_true(array_equal(ary.ravel(order='F'), collectedVals))

        # check that packing returns original array
        assert_true(array_equal(ary, seriesAry))

    def test_toSeriesWithSplitsAndPack(self):
        strategy = SimpleBlockingStrategy((1, 2), units="s")
        self._run_tst_toSeriesWithSplitsAndPack(strategy)

    def test_toSeriesWithPaddedSplitsAndPack(self):
        strategy = PaddedBlockingStrategy((1, 2), units="s", padding=(1, 1))
        self._run_tst_toSeriesWithSplitsAndPack(strategy)

    def test_toSeriesWithInefficientSplitAndSortedPack(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((4, 2))

        image = ImagesLoader(self.sc).fromArrays(ary)
        series = image.toBlocks((2, 1), units="s").toSeries()

        seriesVals = series.collect()
        seriesAry = series.pack(sorting=True)

        # check ordering of keys
        assert_equals((0, 0), seriesVals[0][0])  # first key
        assert_equals((1, 0), seriesVals[1][0])  # second key
        assert_equals((0, 1), seriesVals[2][0])
        assert_equals((1, 1), seriesVals[3][0])
        # end of first block
        # beginning of second block
        assert_equals((2, 0), seriesVals[4][0])
        assert_equals((3, 0), seriesVals[5][0])
        assert_equals((2, 1), seriesVals[6][0])
        assert_equals((3, 1), seriesVals[7][0])

        # check dimensions tuple matches numpy shape
        assert_equals(ary.shape, series.dims.count)

        # check that values are in expected order
        collectedVals = array([kv[1] for kv in seriesVals], dtype=dtypeFunc('int16')).ravel()
        assert_true(array_equal(ary[:2, :].ravel(order='F'), collectedVals[:4]))  # first block
        assert_true(array_equal(ary[2:4, :].ravel(order='F'), collectedVals[4:]))  # second block

        # check that packing returns original array (after sort)
        assert_true(array_equal(ary, seriesAry))

    def test_toBlocksWithSplit(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))

        image = ImagesLoader(self.sc).fromArrays(ary)
        groupedblocks = image.toBlocks((1, 2), units="s")

        # collectedblocks = blocks.collect()
        collectedgroupedblocks = groupedblocks.collect()
        assert_equals((0, 0), collectedgroupedblocks[0][0].spatialKey)
        assert_true(array_equal(ary[:, :2].ravel(), collectedgroupedblocks[0][1].ravel()))
        assert_equals((0, 2), collectedgroupedblocks[1][0].spatialKey)
        assert_true(array_equal(ary[:, 2:].ravel(), collectedgroupedblocks[1][1].ravel()))

    def test_toSeriesBySlices(self):
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)

        imageData = ImagesLoader(self.sc).fromArrays(arys)
        imageData.cache()

        testParams = [
            (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3),
            (1, 3, 1), (1, 3, 2), (1, 3, 3),
            (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 1), (2, 2, 2), (2, 2, 3),
            (2, 3, 1), (2, 3, 2), (2, 3, 3)]

        for bpd in testParams:
            series = imageData.toBlocks(bpd, units="s").toSeries().collect()
            self.evaluateSeries(arys, series, sz)

    def _run_tst_roundtripThroughBlocks(self, strategy):
        imagepath = findSourceTreeDir("utils/data/fish/images")
        images = ImagesLoader(self.sc).fromTif(imagepath)
        blockedimages = images.toBlocks(strategy)
        recombinedimages = blockedimages.toImages()

        collectedimages = images.collect()
        roundtrippedimages = recombinedimages.collect()
        for orig, roundtripped in zip(collectedimages, roundtrippedimages):
            assert_true(array_equal(orig[1], roundtripped[1]))

    def test_roundtripThroughBlocks(self):
        strategy = SimpleBlockingStrategy((2, 2, 2), units="s")
        self._run_tst_roundtripThroughBlocks(strategy)

    def test_roundtripThroughPaddedBlocks(self):
        strategy = PaddedBlockingStrategy((2, 2, 2), units="s", padding=2)
        self._run_tst_roundtripThroughBlocks(strategy)


class TestImagesMethods(PySparkTestCase):

    @staticmethod
    def _run_maxProject(image, inputArys, axis):
        from numpy import amax
        data = image.maxProjection(axis=axis)
        expectedArys = map(lambda ary: amax(ary, axis=axis), inputArys)
        return data, expectedArys

    @staticmethod
    def _run_maxminProject(image, inputArys, axis):
        from numpy import amax, amin
        data = image.maxminProjection(axis=axis)
        expectedArys = map(lambda ary: amax(ary, axis=axis) + amin(ary, axis=axis), inputArys)
        return data, expectedArys

    def _run_tst_maxProject(self, runFcn):
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)

        imageData = ImagesLoader(self.sc).fromArrays(arys)
        for ax in xrange(arys[0].ndim):
            projectedData, expectedArys = runFcn(imageData, arys, ax)
            maxProjected = projectedData.collect()
            for actual, expected in zip(maxProjected, expectedArys):
                assert_true(array_equal(expected, actual[1]))

            expectedShape = list(arys[0].shape)
            del expectedShape[ax]
            assert_equals(tuple(expectedShape), maxProjected[0][1].shape)
            assert_equals(tuple(expectedShape), projectedData._dims.count)
            assert_equals(str(arys[0].dtype), str(maxProjected[0][1].dtype))
            assert_equals(str(maxProjected[0][1].dtype), projectedData._dtype)

    def test_maxProjection(self):
        self._run_tst_maxProject(TestImagesMethods._run_maxProject)

    def test_maxminProjection(self):
        self._run_tst_maxProject(TestImagesMethods._run_maxminProject)

    def test_subsample(self):
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)
        sampFactors = [2, (2, 3, 3)]

        def subsamp(ary, factor):
            if not hasattr(factor, "__len__"):
                factor = [factor] * ary.ndim

            slices = [slice(0, ary.shape[i], factor[i]) for i in xrange(ary.ndim)]
            return ary[slices]

        imageData = ImagesLoader(self.sc).fromArrays(arys)
        for sampFactor in sampFactors:
            subsampData = imageData.subsample(sampFactor)
            expectedArys = map(lambda ary: subsamp(ary, sampFactor), arys)
            subsampled = subsampData.collect()
            for actual, expected in zip(subsampled, expectedArys):
                assert_true(array_equal(expected, actual[1]))

            assert_equals(tuple(expectedArys[0].shape), subsampled[0][1].shape)
            assert_equals(tuple(expectedArys[0].shape), subsampData._dims.count)
            assert_equals(str(arys[0].dtype), str(subsampled[0][1].dtype))
            assert_equals(str(subsampled[0][1].dtype), subsampData._dtype)

    @staticmethod
    def _run_filter(ary, filterFunc, radius):
        if ary.ndim <= 2 or size(radius) > 1:
            return filterFunc(ary, radius)
        else:
            cpy = zeros(ary.shape, dtype=ary.dtype)
            for z in xrange(ary.shape[-1]):
                slices = [slice(None)] * (ary.ndim-1) + [slice(z, z+1, 1)]
                cpy[slices] = filterFunc(ary[slices], radius)
            return cpy

    def _run_tst_filter(self, dataFunc, filterFunc):
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)
        sigma = 2

        imageData = ImagesLoader(self.sc).fromArrays(arys)
        filteredData = dataFunc(imageData, sigma)
        filtered = filteredData.collect()
        expectedArys = map(lambda ary: TestImagesMethods._run_filter(ary, filterFunc, sigma), arys)
        for actual, expected in zip(filtered, expectedArys):
            assert_true(allclose(expected, actual[1]))

        assert_equals(tuple(expectedArys[0].shape), filtered[0][1].shape)
        assert_equals(tuple(expectedArys[0].shape), filteredData._dims.count)
        assert_equals(str(arys[0].dtype), str(filtered[0][1].dtype))
        assert_equals(str(filtered[0][1].dtype), filteredData._dtype)

    def _run_tst_filter_3d_sigma(self, dataFunc, filterFunc):
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)
        sigma = [2, 2, 2]

        imageData = ImagesLoader(self.sc).fromArrays(arys)
        filteredData = dataFunc(imageData, sigma)
        filtered = filteredData.collect()
        expectedArys = map(lambda ary: TestImagesMethods._run_filter(ary, filterFunc, sigma), arys)
        for actual, expected in zip(filtered, expectedArys):
            assert_true(allclose(expected, actual[1]))

        assert_equals(tuple(expectedArys[0].shape), filtered[0][1].shape)
        assert_equals(tuple(expectedArys[0].shape), filteredData._dims.count)
        assert_equals(str(arys[0].dtype), str(filtered[0][1].dtype))
        assert_equals(str(filtered[0][1].dtype), filteredData._dtype)

    def test_gaussFilter3d(self):
        from scipy.ndimage.filters import gaussian_filter
        from thunder.rdds.images import Images
        self._run_tst_filter(Images.gaussianFilter, gaussian_filter)
        self._run_tst_filter_3d_sigma(Images.gaussianFilter, gaussian_filter)

    def test_medianFilter3d(self):
        from scipy.ndimage.filters import median_filter
        from thunder.rdds.images import Images
        self._run_tst_filter(Images.medianFilter, median_filter)
        self._run_tst_filter_3d_sigma(Images.medianFilter, median_filter)

    def test_uniformFilter3d(self):
        from scipy.ndimage.filters import uniform_filter
        from thunder.rdds.images import Images
        self._run_tst_filter(Images.uniformFilter, uniform_filter)
        self._run_tst_filter_3d_sigma(Images.uniformFilter, uniform_filter)

    def _run_tst_crop(self, minBounds, maxBounds):
        dims = (2, 2, 4)
        sz = reduce(lambda x, y: x*y, dims)
        origAry = arange(sz, dtype=dtypeFunc('int16')).reshape(dims)
        imageData = ImagesLoader(self.sc).fromArrays([origAry])
        croppedData = imageData.crop(minBounds, maxBounds)
        crop = croppedData.collect()[0][1]

        slices = []
        for minb, maxb in zip(minBounds, maxBounds):
            # skip the bounds-checking that we do in actual function; assume minb is <= maxb
            if minb < maxb:
                slices.append(slice(minb, maxb))
            else:
                slices.append(minb)

        expected = squeeze(origAry[slices])
        assert_true(array_equal(expected, crop))
        assert_equals(tuple(expected.shape), croppedData._dims.count)
        assert_equals(str(expected.dtype), croppedData._dtype)

    def test_crop(self):
        self._run_tst_crop((0, 0, 0), (2, 2, 2))

    def test_cropAndSqueeze(self):
        self._run_tst_crop((0, 0, 1), (2, 2, 1))

    def test_planes(self):
        dims = (2, 2, 4)
        sz = reduce(lambda x, y: x*y, dims)
        origAry = arange(sz, dtype=dtypeFunc('int16')).reshape(dims)
        imageData = ImagesLoader(self.sc).fromArrays([origAry])
        planedData = imageData.planes(0, 2)
        planed = planedData.collect()[0][1]

        expected = squeeze(origAry[slice(None), slice(None), slice(0, 2)])
        assert_true(array_equal(expected, planed))
        assert_equals(tuple(expected.shape), planedData._dims.count)
        assert_equals(str(expected.dtype), planedData._dtype)

    def test_subtract(self):
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)
        subVals = [1, arange(sz, dtype=dtypeFunc('int16')).reshape(sh)]

        imageData = ImagesLoader(self.sc).fromArrays(arys)
        for subVal in subVals:
            subData = imageData.subtract(subVal)
            subtracted = subData.collect()
            expectedArys = map(lambda ary: ary - subVal, arys)
            for actual, expected in zip(subtracted, expectedArys):
                assert_true(allclose(expected, actual[1]))


class TestImagesStats(PySparkTestCase):
    def test_mean(self):
        from test_utils import elementwiseMean
        arys, shape, size = _generateTestArrays(2, 'uint8')
        imageData = ImagesLoader(self.sc).fromArrays(arys)
        meanVal = imageData.mean()
        expected = elementwiseMean(arys).astype('float16')
        assert_true(allclose(expected, meanVal))
        assert_equals('float64', str(meanVal.dtype))

    def test_sum(self):
        from numpy import add
        arys, shape, size = _generateTestArrays(2, 'uint8')
        imageData = ImagesLoader(self.sc).fromArrays(arys)
        sumVal = imageData.sum(dtype='uint32')

        arys = [ary.astype('uint32') for ary in arys]
        expected = reduce(add, arys)
        assert_true(array_equal(expected, sumVal))
        assert_equals('uint32', str(sumVal.dtype))

    def test_variance(self):
        from test_utils import elementwiseVar
        arys, shape, size = _generateTestArrays(2, 'uint8')
        imageData = ImagesLoader(self.sc).fromArrays(arys)
        varVal = imageData.variance()

        expected = elementwiseVar([ary.astype('float16') for ary in arys])
        assert_true(allclose(expected, varVal))
        assert_equals('float64', str(varVal.dtype))

    def test_stdev(self):
        from test_utils import elementwiseStdev
        arys, shape, size = _generateTestArrays(2, 'uint8')
        imageData = ImagesLoader(self.sc).fromArrays(arys)
        stdval = imageData.stdev()

        expected = elementwiseStdev([ary.astype('float16') for ary in arys])
        assert_true(allclose(expected, stdval))
        assert_equals('float64', str(stdval.dtype))

    def test_stats(self):
        from test_utils import elementwiseMean, elementwiseVar
        arys, shape, size = _generateTestArrays(2, 'uint8')
        imageData = ImagesLoader(self.sc).fromArrays(arys)
        statsval = imageData.stats()

        floatarys = [ary.astype('float16') for ary in arys]
        # StatsCounter contains a few different measures, only test a couple:
        expectedMean = elementwiseMean(floatarys)
        expectedVar = elementwiseVar(floatarys)
        assert_true(allclose(expectedMean, statsval.mean()))
        assert_true(allclose(expectedVar, statsval.variance()))

    def test_max(self):
        from numpy import maximum
        arys, shape, size = _generateTestArrays(2, 'uint8')
        imageData = ImagesLoader(self.sc).fromArrays(arys)
        maxVal = imageData.max()
        assert_true(array_equal(reduce(maximum, arys), maxVal))

    def test_min(self):
        from numpy import minimum
        arys, shape, size = _generateTestArrays(2, 'uint8')
        imageData = ImagesLoader(self.sc).fromArrays(arys)
        minVal = imageData.min()
        assert_true(array_equal(reduce(minimum, arys), minVal))


class TestImagesMeanByRegions(PySparkTestCase):
    def setUp(self):
        super(TestImagesMeanByRegions, self).setUp()
        self.ary1 = array([[3, 5], [6, 8]], dtype='int32')
        self.ary2 = array([[13, 15], [16, 18]], dtype='int32')
        self.images = ImagesLoader(self.sc).fromArrays([self.ary1, self.ary2])

    def __checkAttrPropagation(self, newImages, newDims):
        assert_equals(newDims, newImages._dims.count)
        assert_equals(self.images._nrecords, newImages._nrecords)
        assert_equals(self.images._dtype, newImages._dtype)

    def test_badMaskShapeThrowsValueError(self):
        mask = array([[1]], dtype='int16')
        assert_raises(ValueError, self.images.meanByRegions, mask)

    def test_meanWithFloatMask(self):
        mask = array([[1.0, 0.0], [0.0, 1.0]], dtype='float32')
        regionMeanImages = self.images.meanByRegions(mask)
        self.__checkAttrPropagation(regionMeanImages, (1, 1))
        collected = regionMeanImages.collect()
        assert_equals(2, len(collected))
        assert_equals((1, 1), collected[0][1].shape)
        # check keys
        assert_equals(0, collected[0][0])
        assert_equals(1, collected[1][0])
        # check values
        assert_equals(5, collected[0][1][0])
        assert_equals(15, collected[1][1][0])

    def test_meanWithIntMask(self):
        mask = array([[1, 0], [2, 1]], dtype='uint8')
        regionMeanImages = self.images.meanByRegions(mask)
        self.__checkAttrPropagation(regionMeanImages, (1, 2))
        collected = regionMeanImages.collect()
        assert_equals(2, len(collected))
        assert_equals((1, 2), collected[0][1].shape)
        # check keys
        assert_equals(0, collected[0][0])
        assert_equals(1, collected[1][0])
        # check values
        assert_equals(5, collected[0][1].flat[0])
        assert_equals(6, collected[0][1].flat[1])
        assert_equals(15, collected[1][1].flat[0])
        assert_equals(16, collected[1][1].flat[1])

    def test_meanWithSingleRegionIndices(self):
        indices = [[(1, 1), (0, 0)]]  # one region with two indices
        regionMeanImages = self.images.meanByRegions(indices)
        self.__checkAttrPropagation(regionMeanImages, (1, 1))
        collected = regionMeanImages.collect()
        assert_equals(2, len(collected))
        assert_equals((1, 1), collected[0][1].shape)
        # check keys
        assert_equals(0, collected[0][0])
        assert_equals(1, collected[1][0])
        # check values
        assert_equals(5, collected[0][1][0])
        assert_equals(15, collected[1][1][0])

    def test_meanWithMultipleRegionIndices(self):
        indices = [[(0, 0), (0, 1)], [(0, 1), (1, 0)]]  # two regions with two indices each
        regionMeanImages = self.images.meanByRegions(indices)
        self.__checkAttrPropagation(regionMeanImages, (1, 2))
        collected = regionMeanImages.collect()
        assert_equals(2, len(collected))
        assert_equals((1, 2), collected[0][1].shape)
        # check keys
        assert_equals(0, collected[0][0])
        assert_equals(1, collected[1][0])
        # check values
        assert_equals(4, collected[0][1].flat[0])
        assert_equals(5, collected[0][1].flat[1])
        assert_equals(14, collected[1][1].flat[0])
        assert_equals(15, collected[1][1].flat[1])

    def test_badIndexesThrowErrors(self):
        indices = [[(0, 0), (-1, 0)]]  # index too small (-1)
        assert_raises(ValueError, self.images.meanByRegions, indices)

        indices = [[(0, 0), (2, 0)]]  # index too large (2)
        assert_raises(ValueError, self.images.meanByRegions, indices)

        indices = [[(0, 0), (0,)]]  # too few indices
        assert_raises(ValueError, self.images.meanByRegions, indices)

        indices = [[(0, 0), (0, 1, 0)]]  # too many indices
        assert_raises(ValueError, self.images.meanByRegions, indices)

    def test_meanWithSingleRegionIndices3D(self):
        ary1 = array([[[3, 5, 3], [6, 8, 6]], [[3, 5, 3], [6, 8, 6]]], dtype='int32')
        ary2 = array([[[13, 15, 13], [16, 18, 16]], [[13, 15, 13], [16, 18, 16]]], dtype='int32')
        images = ImagesLoader(self.sc).fromArrays([ary1, ary2])
        indices = [[(1, 1, 1), (0, 0, 0)]]  # one region with two indices
        regionMeanImages = images.meanByRegions(indices)
        self.__checkAttrPropagation(regionMeanImages, (1, 1))
        collected = regionMeanImages.collect()
        # check values
        assert_equals(5, collected[0][1][0])
        assert_equals(15, collected[1][1][0])

class TestImagesLocalCorr(PySparkTestCase):
    """Test accuracy for local correlation
    by comparison to known result
    (verified by directly computing
    result with numpy's mean and corrcoef)

    Test with indexing from both 0 and 1,
    and for both 2D and 3D data
    """

    def get_local_corr(self, data, neighborhood, images=False):
        rdd = self.sc.parallelize(data)
        imgs = Images(rdd) if images else Series(rdd).toImages()
        return imgs.localCorr(neighborhood=neighborhood)

    def test_localCorr_0Indexing_2D(self):

        dataLocal = [
            ((0, 0), array([1.0, 2.0, 3.0])),
            ((0, 1), array([2.0, 2.0, 4.0])),
            ((0, 2), array([9.0, 2.0, 1.0])),
            ((1, 0), array([5.0, 2.0, 5.0])),
            ((2, 0), array([4.0, 2.0, 6.0])),
            ((1, 1), array([4.0, 2.0, 8.0])),
            ((1, 2), array([5.0, 4.0, 1.0])),
            ((2, 1), array([6.0, 3.0, 2.0])),
            ((2, 2), array([0.0, 2.0, 1.0]))
        ]

        # get ground truth by correlating mean with the center
        ts = map(lambda x: x[1], dataLocal)
        mn = mean(ts, axis=0)
        truth = corrcoef(mn, array([4.0, 2.0, 8.0]))[0, 1]

        corr = self.get_local_corr(dataLocal, 1)

        assert(allclose(corr[1][1], truth))

    def test_localCorr_0Indexing_3D(self):

        dataLocal = [
            ((0, 0, 0), array([1.0, 2.0, 3.0])),
            ((0, 1, 0), array([2.0, 2.0, 4.0])),
            ((0, 2, 0), array([9.0, 2.0, 1.0])),
            ((1, 0, 0), array([5.0, 2.0, 5.0])),
            ((2, 0, 0), array([4.0, 2.0, 6.0])),
            ((1, 1, 0), array([4.0, 2.0, 8.0])),
            ((1, 2, 0), array([5.0, 4.0, 1.0])),
            ((2, 1, 0), array([6.0, 3.0, 2.0])),
            ((2, 2, 0), array([0.0, 2.0, 1.0]))
        ]

        # get ground truth by correlating mean with the center
        ts = map(lambda x: x[1], dataLocal)
        mn = mean(ts, axis=0)
        truth = corrcoef(mn, array([4.0, 2.0, 8.0]))[0, 1]

        corr = self.get_local_corr(dataLocal, 1)

        assert(allclose(corr[1][1], truth))

    def test_localCorr_Images_2D(self):

        dataLocal = [
            (0, array([[1.0, 2.0, 9.0], [5.0, 4.0, 5.0], [4.0, 6.0, 0.0]])),
            (1, array([[2.0, 2.0, 2.0], [2.0, 2.0, 4.0], [2.0, 3.0, 2.0]])),
            (2, array([[3.0, 4.0, 1.0], [5.0, 8.0, 1.0], [6.0, 2.0, 1.0]]))
        ]

        from scipy.ndimage.filters import uniform_filter
        imgs = map(lambda x: x[1], dataLocal)
        # Blur each image and extract the center pixel
        mn = map(lambda img: uniform_filter(img, 3)[1, 1], imgs)
        truth = corrcoef(mn, array([4.0, 2.0, 8.0]))[0, 1]

        corr = self.get_local_corr(dataLocal, 1, images=True)

        assert(allclose(corr[1][1], truth))

    def test_localCorr_Images_3D(self):

        dataLocal = [
            (0, array([[1.0, 2.0, 9.0], [5.0, 4.0, 5.0], [4.0, 6.0, 0.0]])),
            (1, array([[2.0, 2.0, 2.0], [2.0, 2.0, 4.0], [2.0, 3.0, 2.0]])),
            (2, array([[3.0, 4.0, 1.0], [5.0, 8.0, 1.0], [6.0, 2.0, 1.0]]))
        ]

        from scipy.ndimage.filters import uniform_filter
        imgs = map(lambda x: x[1], dataLocal)
        # Blur each image and extract the center pixel
        mn = map(lambda img: uniform_filter(img, 3)[1, 1], imgs)
        truth = corrcoef(mn, array([4.0, 2.0, 8.0]))[0, 1]

        corr = self.get_local_corr(dataLocal, 1, images=True)

        assert(allclose(corr[1][1], truth))


class TestImagesUsingOutputDir(PySparkTestCaseWithOutputDir):

    @staticmethod
    def _findSourceTreeDir(dirname="utils/data"):
        testDirPath = os.path.dirname(os.path.realpath(__file__))
        testResourcesDirPath = os.path.join(testDirPath, "..", "thunder", dirname)
        if not os.path.isdir(testResourcesDirPath):
            raise IOError("Directory "+testResourcesDirPath+" not found")
        return testResourcesDirPath

    def _run_tstSaveAsBinarySeries(self, testIdx, narys_, valDtype, groupingDim_):
        """Pseudo-parameterized test fixture, allows reusing existing spark context
        """
        paramStr = "(groupingdim=%d, valuedtype='%s')" % (groupingDim_, valDtype)
        arys, aryShape, arySize = _generateTestArrays(narys_, dtype_=valDtype)
        dims = aryShape[:]
        outdir = os.path.join(self.outputdir, "anotherdir%02d" % testIdx)

        images = ImagesLoader(self.sc).fromArrays(arys)

        slicesPerDim = [1]*arys[0].ndim
        slicesPerDim[groupingDim_] = arys[0].shape[groupingDim_]
        images.toBlocks(slicesPerDim, units="splits").saveAsBinarySeries(outdir)

        ndims = len(aryShape)
        # prevent padding to 4-byte boundaries: "=" specifies no alignment
        unpacker = struct.Struct('=' + 'h'*ndims + dtypeFunc(valDtype).char*narys_)

        def calcExpectedNKeys():
            tmpShape = list(dims[:])
            del tmpShape[groupingDim_]
            return prod(tmpShape)
        expectedNKeys = calcExpectedNKeys()

        def byrec(f_, unpacker_, nkeys_):
            rec = True
            while rec:
                rec = f_.read(unpacker_.size)
                if rec:
                    allRecVals = unpacker_.unpack(rec)
                    yield allRecVals[:nkeys_], allRecVals[nkeys_:]

        outFilenames = glob.glob(os.path.join(outdir, "*.bin"))
        assert_equals(dims[groupingDim_], len(outFilenames))
        for outFilename in outFilenames:
            with open(outFilename, 'rb') as f:
                nkeys = 0
                for keys, vals in byrec(f, unpacker, ndims):
                    nkeys += 1
                    assert_equals(narys_, len(vals))
                    for valIdx, val in enumerate(vals):
                        assert_equals(arys[valIdx][keys], val, "Expected %g, got %g, for test %d %s" %
                                      (arys[valIdx][keys], val, testIdx, paramStr))
                assert_equals(expectedNKeys, nkeys)

        confName = os.path.join(outdir, "conf.json")
        assert_true(os.path.isfile(confName))
        with open(os.path.join(outdir, "conf.json"), 'r') as fconf:
            import json
            conf = json.load(fconf)
            assert_equals(outdir, conf['input'])
            assert_equals(len(aryShape), conf['nkeys'])
            assert_equals(narys_, conf['nvalues'])
            assert_equals(valDtype, conf['valuetype'])
            assert_equals('int16', conf['keytype'])

        assert_true(os.path.isfile(os.path.join(outdir, 'SUCCESS')))

    def test_saveAsBinarySeries(self):
        narys = 3
        arys, aryShape, _ = _generateTestArrays(narys)

        outdir = os.path.join(self.outputdir, "anotherdir")
        os.mkdir(outdir)
        assert_raises(ValueError, ImagesLoader(self.sc).fromArrays(arys).toBlocks((1, 1, 1), units="s")
                      .saveAsBinarySeries, outdir)

        groupingDims = xrange(len(aryShape))
        dtypes = ('int16', 'int32', 'float32')
        paramIters = itertools.product(groupingDims, dtypes)

        for idx, params in enumerate(paramIters):
            gd, dt = params
            self._run_tstSaveAsBinarySeries(idx, narys, dt, gd)

    def _run_tst_roundtripConvertToSeries(self, images, strategy):
        outdir = os.path.join(self.outputdir, "fish-series-dir")

        partitionedimages = images.toBlocks(strategy)
        series = partitionedimages.toSeries()
        series_ary = series.pack()

        partitionedimages.saveAsBinarySeries(outdir)
        converted_series = SeriesLoader(self.sc).fromBinary(outdir)
        converted_series_ary = converted_series.pack()

        assert_equals(images.dims.count, series.dims.count)
        expected_shape = tuple([images.nrecords] + list(images.dims.count))
        assert_equals(expected_shape, series_ary.shape)
        assert_true(array_equal(series_ary, converted_series_ary))

    def test_roundtripConvertToSeries(self):
        imagepath = findSourceTreeDir("utils/data/fish/images")

        images = ImagesLoader(self.sc).fromTif(imagepath)
        strategy = SimpleBlockingStrategy.generateFromBlockSize(images, blockSize=76 * 20)
        self._run_tst_roundtripConvertToSeries(images, strategy)

    def test_fromStackToSeriesWithPack(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))
        filename = os.path.join(self.outputdir, "test.stack")
        ary.tofile(filename)

        image = ImagesLoader(self.sc).fromStack(filename, dims=(4, 2))
        strategy = SimpleBlockingStrategy.generateFromBlockSize(image, "150M")
        series = image.toBlocks(strategy).toSeries()

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

    def test_saveAsBinaryImages(self):
        narys = 3
        arys, aryShape, _ = _generateTestArrays(narys)

        outdir = os.path.join(self.outputdir, "binary-images")

        images = ImagesLoader(self.sc).fromArrays(arys)
        images.saveAsBinaryImages(outdir)

        outFilenames = sorted(glob.glob(os.path.join(outdir, "*.bin")))
        trueFilenames = map(lambda f: os.path.join(outdir, f),
                            ['image-00000.bin', 'image-00001.bin', 'image-00002.bin'])
        assert_true(os.path.isfile(os.path.join(outdir, 'SUCCESS')))
        assert_true(os.path.isfile(os.path.join(outdir, "conf.json")))
        assert_equals(outFilenames, trueFilenames)

    def test_saveAsBinaryImagesRoundtrip(self):

        def roundTrip(images, dtype):
            outdir = os.path.join(self.outputdir, "binary-images-" + dtype)
            images.astype(dtype).saveAsBinaryImages(outdir)
            newimages = ImagesLoader(self.sc).fromStack(outdir, ext='bin')
            array_equal(images.first()[1], newimages.first()[1])

        narys = 3
        arys, aryShape, _ = _generateTestArrays(narys)
        images = ImagesLoader(self.sc).fromArrays(arys)

        map(lambda d: roundTrip(images, d), ['int16', 'int32', 'float64'])

if __name__ == "__main__":
    if not _have_image:
        print "NOTE: Skipping PIL/pillow tests as neither seem to be installed and functional"
    unittest.main()
    if not _have_image:
        print "NOTE: PIL/pillow tests were skipped as neither seem to be installed and functional"

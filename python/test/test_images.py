from collections import Counter
import glob
import struct
import os
from operator import mul
from numpy import allclose, arange, array, array_equal, prod, squeeze, zeros
from numpy import dtype as dtypeFunc
import itertools
from nose.tools import assert_equals, assert_true, assert_almost_equal, assert_raises

from thunder.rdds.fileio.imagesloader import ImagesLoader
from thunder.rdds.fileio.seriesloader import SeriesLoader
from thunder.rdds.images import _BlockMemoryAsReversedSequence
from test_utils import *

_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    Image = None


def _generateTestArrays(narys, dtype_='int16'):
    sh = 4, 3, 3
    sz = prod(sh)
    arys = [arange(i, i+sz, dtype=dtypeFunc(dtype_)).reshape(sh) for i in xrange(0, sz * narys, sz)]
    return arys, sh, sz


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
        series = imageData.toSeries(groupingDim=0).collect()

        self.evaluateSeries(arys, series, sz)

    def test_toSeriesWithPack(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))

        image = ImagesLoader(self.sc).fromArrays(ary)
        series = image.toSeries()

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
        series = image.toSeries()

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

    def test_toSeriesWithSplitsAndPack(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((4, 2))

        image = ImagesLoader(self.sc).fromArrays(ary)
        series = image.toSeries(splitsPerDim=(1, 2))

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

    def test_toSeriesWithInefficientSplitAndSortedPack(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((4, 2))

        image = ImagesLoader(self.sc).fromArrays(ary)
        series = image.toSeries(splitsPerDim=(2, 1))

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
        blocks = image._scatterToBlocks(blocksPerDim=(1, 2))
        groupedBlocks = blocks._groupIntoSeriesBlocks()

        # collectedblocks = blocks.collect()
        collectedGroupedBlocks = groupedBlocks.collect()
        assert_equals((0, 0), collectedGroupedBlocks[0][0])
        assert_true(array_equal(ary[:, :2].ravel(), collectedGroupedBlocks[0][1].values.ravel()))
        assert_equals((0, 2), collectedGroupedBlocks[1][0])
        assert_true(array_equal(ary[:, 2:].ravel(), collectedGroupedBlocks[1][1].values.ravel()))

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
            series = imageData.toSeries(splitsPerDim=bpd).collect()

            self.evaluateSeries(arys, series, sz)

    def test_toBlocksBySlices(self):
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)

        imageData = ImagesLoader(self.sc).fromArrays(arys)

        testParams = [
            (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3),
            (1, 3, 1), (1, 3, 2), (1, 3, 3),
            (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 1), (2, 2, 2), (2, 2, 3),
            (2, 3, 1), (2, 3, 2), (2, 3, 3)]
        for bpd in testParams:
            blocks = imageData._toBlocksBySplits(bpd).collect()

            expectedNUniqueKeys = reduce(mul, bpd)
            expectedValsPerKey = narys

            keysToCounts = Counter([kv[0] for kv in blocks])
            assert_equals(expectedNUniqueKeys, len(keysToCounts))
            assert_equals([expectedValsPerKey] * expectedNUniqueKeys, keysToCounts.values())

            gatheredAry = None
            for _, block in blocks:
                if gatheredAry is None:
                    gatheredAry = zeros(block.origshape, dtype='int16')
                gatheredAry[block.origslices] = block.values

            for i in xrange(narys):
                assert_true(array_equal(arys[i], gatheredAry[i]))

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
        self._run_tst_maxProject(TestImages._run_maxProject)

    def test_maxminProjection(self):
        self._run_tst_maxProject(TestImages._run_maxminProject)

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
        if ary.ndim <= 2:
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
        expectedArys = map(lambda ary: TestImages._run_filter(ary, filterFunc, sigma), arys)
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

    def test_medianFilter3d(self):
        from scipy.ndimage.filters import median_filter
        from thunder.rdds.images import Images
        self._run_tst_filter(Images.medianFilter, median_filter)

    def test_planes(self):
        # params are images shape, bottom, top, inclusize, expected slices of orig ary
        PARAMS = [((2, 2, 4), 1, 2, True, [slice(None), slice(None), slice(1, 3)]),
                  ((2, 2, 4), 0, 2, False, [slice(None), slice(None), slice(1, 2)])]
        for params in PARAMS:
            sz = reduce(lambda x, y: x*y, params[0])
            origAry = arange(sz, dtype='int16').reshape(params[0])
            imageData = ImagesLoader(self.sc).fromArrays([origAry])
            planedData = imageData.planes(params[1], params[2], params[3])
            planed = planedData.collect()

            expected = squeeze(origAry[params[4]])
            assert_true(array_equal(expected, planed[0][1]))
            assert_equals(tuple(expected.shape), planedData._dims.count)
            assert_equals(str(expected.dtype), planedData._dtype)

    def test_subtract(self):
        narys = 3
        arys, sh, sz = _generateTestArrays(narys)
        subVals = [1, arange(sz, dtype='int16').reshape(sh)]

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
        assert_equals('float16', str(meanVal.dtype))

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
        assert_equals('float16', str(varVal.dtype))

    def test_stdev(self):
        from test_utils import elementwiseStdev
        arys, shape, size = _generateTestArrays(2, 'uint8')
        imageData = ImagesLoader(self.sc).fromArrays(arys)
        stdval = imageData.stdev()

        expected = elementwiseStdev([ary.astype('float16') for ary in arys])
        assert_true(allclose(expected, stdval))
        #assert_equals('float16', str(stdval.dtype))
        # it isn't clear to me why this comes out as float32 and not float16, especially
        # given that var returns float16, as expected. But I'm not too concerned about it.
        # Consider this documentation of current behavior rather than a description of
        # desired behavior.
        assert_equals('float32', str(stdval.dtype))

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

        images.saveAsBinarySeries(outdir, groupingDim=groupingDim_)

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
            assert_equals(tuple(dims), tuple(conf['dims']))
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
        assert_raises(ValueError, ImagesLoader(self.sc).fromArrays(arys).saveAsBinarySeries, outdir, 0)

        groupingDims = xrange(len(aryShape))
        dtypes = ('int16', 'int32', 'float32')
        paramIters = itertools.product(groupingDims, dtypes)

        for idx, params in enumerate(paramIters):
            gd, dt = params
            self._run_tstSaveAsBinarySeries(idx, narys, dt, gd)

    def test_roundtripConvertToSeries(self):
        imagepath = TestImagesUsingOutputDir._findSourceTreeDir("utils/data/fish/tif-stack")
        outdir = os.path.join(self.outputdir, "fish-series-dir")

        images = ImagesLoader(self.sc).fromMultipageTif(imagepath)
        series = images.toSeries(blockSize=76*20)
        seriesAry = series.pack()

        images.saveAsBinarySeries(outdir, blockSize=76*20)
        convertedSeries = SeriesLoader(self.sc).fromBinary(outdir)
        convertedSeriesAry = convertedSeries.pack()

        assert_equals((76, 87, 2), series.dims.count)
        assert_equals((20, 76, 87, 2), seriesAry.shape)
        assert_true(array_equal(seriesAry, convertedSeriesAry))

    def test_fromStackToSeriesWithPack(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))
        filename = os.path.join(self.outputdir, "test.stack")
        ary.tofile(filename)

        image = ImagesLoader(self.sc).fromStack(filename, dims=(4, 2))
        series = image.toSeries()

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


class TestBlockMemoryAsSequence(unittest.TestCase):

    def test_range(self):
        dims = (2, 2)
        undertest = _BlockMemoryAsReversedSequence(dims)

        assert_equals(3, len(undertest))
        assert_equals((2, 2), undertest.indToSub(0))
        assert_equals((1, 2), undertest.indToSub(1))
        assert_equals((1, 1), undertest.indToSub(2))
        assert_raises(IndexError, undertest.indToSub, 3)


if __name__ == "__main__":
    if not _have_image:
        print "NOTE: Skipping PIL/pillow tests as neither seem to be installed and functional"
    unittest.main()
    if not _have_image:
        print "NOTE: PIL/pillow tests were skipped as neither seem to be installed and functional"

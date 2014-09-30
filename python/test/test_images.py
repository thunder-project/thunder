from collections import Counter
import glob
import struct
import unittest
import os
from operator import mul
from numpy import ndarray, arange, array, array_equal, concatenate, dtype, prod, zeros
from nose.tools import assert_equals, assert_true, assert_almost_equal, assert_raises
import itertools
from thunder.rdds.images import ImagesLoader, ImageBlockValue
from test_utils import PySparkTestCase, PySparkTestCaseWithOutputDir

_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    pass


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


def _generate_test_arrays(narys, dtype='int16'):
    sh = 4, 3, 3
    sz = prod(sh)
    arys = [arange(i, i+sz, dtype=dtype).reshape(sh) for i in xrange(0, sz * narys, sz)]
    return arys, sh, sz


class TestImages(PySparkTestCase):

    def evaluate_series(self, arys, series, sz):
        assert_equals(sz, len(series))
        for serieskey, seriesval in series:
            expectedval = array([ary[serieskey] for ary in arys], dtype='int16')
            assert_true(array_equal(expectedval, seriesval))

    def test_toSeries(self):
        # create 3 arrays of 4x3x3 images (C-order), containing sequential integers
        narys = 3
        arys, sh, sz = _generate_test_arrays(narys)

        imagedata = ImagesLoader(self.sc).fromArrays(arys)
        series = imagedata.toSeries(groupingDim=0).collect()

        self.evaluate_series(arys, series, sz)

    def test_toSeriesBySlices(self):
        narys = 3
        arys, sh, sz = _generate_test_arrays(narys)

        imagedata = ImagesLoader(self.sc).fromArrays(arys)
        imagedata.cache()

        test_params = [
            (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3),
            (1, 3, 1), (1, 3, 2), (1, 3, 3),
            (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 1), (2, 2, 2), (2, 2, 3),
            (2, 3, 1), (2, 3, 2), (2, 3, 3)]
        for bpd in test_params:
            series = imagedata.toSeries(splitsPerDim=bpd).collect()

            self.evaluate_series(arys, series, sz)

    def test_toBlocksByPlanes(self):
        # create 3 arrays of 4x3x3 images (C-order), containing sequential integers
        narys = 3
        arys, sh, sz = _generate_test_arrays(narys)

        grpdim = 0
        blocks = ImagesLoader(self.sc).fromArrays(arys) \
            ._toBlocksByImagePlanes(groupingDim=grpdim).collect()

        assert_equals(sh[grpdim]*narys, len(blocks))

        keystocounts = Counter([kv[0] for kv in blocks])
        # expected keys are (index, 0, 0) (or (z, y, x)) for index in grouping dimension
        expectedkeys = set((idx, 0, 0) for idx in xrange(sh[grpdim]))
        expectednkeys = sh[grpdim]
        assert_equals(expectednkeys, len(keystocounts))
        # check all expected keys are present:
        assert_true(expectedkeys == set(keystocounts.iterkeys()))
        # check all keys appear the expected number of times (once per input array):
        assert_equals([narys]*expectednkeys, keystocounts.values())

        # check that we can get back the expected planes over time:
        for blockkey, blockplane in blocks:
            tpidx = blockplane.origslices[grpdim].start
            planeidx = blockkey[grpdim]
            expectedplane = arys[tpidx][planeidx, :, :]
            assert_true(array_equal(expectedplane, blockplane.values.squeeze()))

    def test_toBlocksBySlices(self):
        narys = 3
        arys, sh, sz = _generate_test_arrays(narys)

        imagedata = ImagesLoader(self.sc).fromArrays(arys)

        test_params = [
            (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3),
            (1, 3, 1), (1, 3, 2), (1, 3, 3),
            (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 1), (2, 2, 2), (2, 2, 3),
            (2, 3, 1), (2, 3, 2), (2, 3, 3)]
        for bpd in test_params:
            blocks = imagedata._toBlocksBySplits(bpd).collect()

            expectednuniquekeys = reduce(mul, bpd)
            expectedvalsperkey = narys

            keystocounts = Counter([kv[0] for kv in blocks])
            assert_equals(expectednuniquekeys, len(keystocounts))
            assert_equals([expectedvalsperkey] * expectednuniquekeys, keystocounts.values())

            gatheredary = None
            for _, block in blocks:
                if gatheredary is None:
                    gatheredary = zeros(block.origshape, dtype='int16')
                gatheredary[block.origslices] = block.values

            for i in xrange(narys):
                assert_true(array_equal(arys[i], gatheredary[i]))


class TestImagesUsingOutputDir(PySparkTestCaseWithOutputDir):

    def _run_tstSaveAsBinarySeries(self, testidx, narys_, valdtype, groupingdim_):
        """Pseudo-parameterized test fixture, allows reusing existing spark context
        """
        paramstr = "(groupingdim=%d, valuedtype='%s')" % (groupingdim_, valdtype)
        arys, aryshape, arysize = _generate_test_arrays(narys_, dtype=valdtype)
        outdir = os.path.join(self.outputdir, "anotherdir%02d" % testidx)

        images = ImagesLoader(self.sc).fromArrays(arys)

        images.saveAsBinarySeries(outdir, groupingDim=groupingdim_)

        ndims = len(aryshape)
        # prevent padding to 4-byte boundaries: "=" specifies no alignment
        unpacker = struct.Struct('=' + 'h'*ndims + dtype(valdtype).char*narys_)

        def calcExpectedNKeys(aryshape__, groupingdim__):
            tmpshape = list(aryshape__[:])
            del tmpshape[groupingdim__]
            return prod(tmpshape)
        expectednkeys = calcExpectedNKeys(aryshape, groupingdim_)

        def byrec(f_, unpacker_, nkeys_):
            rec = True
            while rec:
                rec = f_.read(unpacker_.size)
                if rec:
                    allrecvals = unpacker_.unpack(rec)
                    yield allrecvals[:nkeys_], allrecvals[nkeys_:]

        outfilenames = glob.glob(os.path.join(outdir, "*.bin"))
        assert_equals(aryshape[groupingdim_], len(outfilenames))
        for outfilename in outfilenames:
            with open(outfilename, 'rb') as f:
                nkeys = 0
                for keys, vals in byrec(f, unpacker, ndims):
                    nkeys += 1
                    assert_equals(narys_, len(vals))
                    for validx, val in enumerate(vals):
                        assert_equals(arys[validx][keys], val, "Expected %g, got %g, for test %d %s" %
                                      (arys[validx][keys], val, testidx, paramstr))
                assert_equals(expectednkeys, nkeys)

        confname = os.path.join(outdir, "conf.json")
        assert_true(os.path.isfile(confname))
        with open(os.path.join(outdir, "conf.json"), 'r') as fconf:
            import json
            conf = json.load(fconf)
            assert_equals(outdir, conf['input'])
            assert_equals(tuple(aryshape), tuple(conf['dims']))
            assert_equals(len(aryshape), conf['nkeys'])
            assert_equals(narys_, conf['nvalues'])
            assert_equals(valdtype, conf['format'])
            assert_equals('int16', conf['keyformat'])

        assert_true(os.path.isfile(os.path.join(outdir, 'SUCCESS')))

    def test_saveAsBinarySeries(self):
        narys = 3
        arys, aryshape, _ = _generate_test_arrays(narys)

        outdir = os.path.join(self.outputdir, "anotherdir")
        os.mkdir(outdir)
        assert_raises(ValueError, ImagesLoader(self.sc).fromArrays(arys).saveAsBinarySeries, outdir, 0)

        groupingdims = xrange(len(aryshape))
        dtypes = ('int16', 'int32', 'float32')
        paramiters = itertools.product(groupingdims, dtypes)

        for idx, params in enumerate(paramiters):
            gd, dt = params
            self._run_tstSaveAsBinarySeries(idx, narys, dt, gd)


class TestImageBlockValue(unittest.TestCase):

    def test_fromArrayByPlane(self):
        values = arange(12, dtype='int16').reshape((3, 4), order='C')

        planedim = 0
        planedimidx = 1
        imageblock = ImageBlockValue.fromArrayByPlane(values, planedim=planedim, planeidx=planedimidx)

        assert_equals(values.shape, imageblock.origshape)
        assert_equals(slice(planedimidx, planedimidx+1, 1), imageblock.origslices[planedim])
        assert_equals(slice(None), imageblock.origslices[1])
        assert_true(array_equal(values[planedimidx, :].flatten(order='C'), imageblock.values.flatten(order='C')))

    def test_fromArrayBySlices(self):
        values = arange(12, dtype='int16').reshape((3, 4), order='C')

        slices = [[slice(0, 3)], [slice(0, 2), slice(2, 4)]]
        slicesiter = itertools.product(*slices)

        imageblocks = [ImageBlockValue.fromArrayBySlices(values, sls) for sls in slicesiter]
        assert_equals(2, len(imageblocks))
        assert_equals((3, 2), imageblocks[0].values.shape)
        assert_true(array_equal(values[(slice(0, 3), slice(0, 2))], imageblocks[0].values))

    def test_fromPlanarBlocks(self):
        values = arange(36, dtype='int16').reshape((3, 4, 3), order='F')

        imageblocks = [ImageBlockValue.fromArrayByPlane(values, -1, i) for i in xrange(values.shape[2])]

        recombblock = ImageBlockValue.fromPlanarBlocks(imageblocks, planarDim=-1)

        assert_true(array_equal(values, recombblock.values))
        assert_equals([slice(None)] * values.ndim, recombblock.origslices)
        assert_equals(values.shape, recombblock.origshape)

    def test_addDimension(self):
        values = arange(12, dtype='int16').reshape((3, 4), order='C')
        morevalues = arange(12, 24, dtype='int16').reshape((3, 4), order='C')

        origshape = values.shape
        origslices = [slice(None)] * values.ndim
        newdimsize = 2
        initimageblock = ImageBlockValue(origshape=origshape, origslices=origslices, values=values)
        anotherinitimageblock = ImageBlockValue(origshape=origshape, origslices=origslices, values=morevalues)

        imageblock = initimageblock.addDimension(newdimidx=0, newdimsize=newdimsize)
        anotherimageblock = anotherinitimageblock.addDimension(newdimidx=1, newdimsize=newdimsize)

        expectedorigshape = tuple([newdimsize] + list(initimageblock.origshape))
        assert_equals(expectedorigshape, imageblock.origshape)
        assert_equals(expectedorigshape, anotherimageblock.origshape)

        expectednslices = len(expectedorigshape)
        assert_equals(expectednslices, len(imageblock.origslices))
        assert_equals(expectednslices, len(anotherimageblock.origslices))

        assert_equals(slice(0, 1, 1), imageblock.origslices[0])
        assert_equals(slice(1, 2, 1), anotherimageblock.origslices[0])

        expectedshape = tuple([1] + list(values.shape))
        assert_equals(expectedshape, imageblock.values.shape)
        assert_equals(expectedshape, anotherimageblock.values.shape)

        # check that straight array concatenation works as expected in this particular case
        expectedcatvals = arange(24, dtype='int16')
        actualcatvals = concatenate((imageblock.values, anotherimageblock.values), axis=0).flatten(order='C')
        assert_true(array_equal(expectedcatvals, actualcatvals))

    def test_toSeriesIter(self):
        sh = 3, 3, 4
        sz = prod(sh)
        imageblock = ImageBlockValue.fromArray(arange(sz, dtype='int16').reshape(sh, order='C'))

        series = list(imageblock.toSeriesIter(-1))

        expectedseries = []
        for n, ij in zip(xrange(0, sz, 4), itertools.product(xrange(3), xrange(3))):
            expectedkv = (ij[0], ij[1]), arange(n, n+4, dtype='int16')
            expectedseries.append(expectedkv)

        for actual, expected in zip(series, expectedseries):
            # check key equality
            assert_equals(expected[0], actual[0])
            # check value equality
            assert_true(array_equal(expected[1], actual[1]))


if __name__ == "__main__":
    if not _have_image:
        print "NOTE: Skipping PIL/pillow tests as neither seem to be installed and functional"
    unittest.main()
    if not _have_image:
        print "NOTE: PIL/pillow tests were skipped as neither seem to be installed and functional"

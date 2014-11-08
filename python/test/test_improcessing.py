import shutil
import tempfile
from test_utils import PySparkTestCase
from numpy import random, allclose, arange
from scipy.ndimage.interpolation import shift
from thunder.improcessing.register import Register
from thunder.rdds.fileio.imagesloader import ImagesLoader


class ImprocessingTestCase(PySparkTestCase):
    def setUp(self):
        super(ImprocessingTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(ImprocessingTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestRegistration(ImprocessingTestCase):

    def test_crosscorr_image(self):

        ref = random.randn(25, 25)

        im = shift(ref, [2, 0], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)
        paramout = Register('crosscorr').estimate(imin, ref)[0][1]
        imout = Register('crosscorr').transform(imin, ref).first()[1]
        assert(allclose(ref[:-2, :], imout[:-2, :]))
        assert(allclose(paramout, [2, 0]))

        im = shift(ref, [0, 2], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)
        paramout = Register('crosscorr').estimate(imin, ref)[0][1]
        imout = Register('crosscorr').transform(imin, ref).first()[1]
        assert(allclose(ref[:, :-2], imout[:, :-2]))
        assert(allclose(paramout, [0, 2]))

        im = shift(ref, [2, -2], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)
        paramout = Register('crosscorr').estimate(imin, ref)[0][1]
        imout = Register('crosscorr').transform(imin, ref).first()[1]
        assert(allclose(ref[:-2, 2:], imout[:-2, 2:]))
        assert(allclose(paramout, [2, -2]))

        im = shift(ref, [-2, 2], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)
        paramout = Register('crosscorr').estimate(imin, ref)[0][1]
        imout = Register('crosscorr').transform(imin, ref).first()[1]
        assert(allclose(ref[2:, :-2], imout[2:, :-2]))
        assert(allclose(paramout, [-2, 2]))

    def test_crosscorr_volume(self):

        ref = random.randn(25, 25, 3)

        im = shift(ref, [2, -2, 0], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)
        paramout = Register('crosscorr').estimate(imin, ref)[0][1]
        imout = Register('crosscorr').transform(imin, ref).first()[1]
        assert(allclose(paramout, [[2, -2], [2, -2], [2, -2]]))
        assert(allclose(ref[:-2, 2:, :], imout[:-2, 2:, :]))

    def test_reference_2d(self):

        im1 = random.randn(25, 25).astype('uint16')
        im2 = random.randn(25, 25).astype('uint16')
        imin = ImagesLoader(self.sc).fromArrays([im1, im2])
        ref = Register.reference(imin)
        assert(allclose(ref, (im1 + im2) / 2))

    def test_reference_3d(self):

        im1 = random.randn(25, 25, 3).astype('uint16')
        im2 = random.randn(25, 25, 3).astype('uint16')
        imin = ImagesLoader(self.sc).fromArrays([im1, im2])
        ref = Register.reference(imin)
        assert(allclose(ref, (im1 + im2) / 2))
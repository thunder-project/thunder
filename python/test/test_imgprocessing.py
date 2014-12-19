import shutil
import tempfile
from test_utils import PySparkTestCase
from numpy import random, allclose, arange
from scipy.ndimage.interpolation import shift
from thunder.imgprocessing.register import Register
from thunder.rdds.fileio.imagesloader import ImagesLoader


class ImgProcessingTestCase(PySparkTestCase):
    def setUp(self):
        super(ImgProcessingTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(ImgProcessingTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestRegistration(ImgProcessingTestCase):

    def test_crosscorr_image(self):

        random.seed(42)
        ref = random.randn(25, 25)

        reg = Register('crosscorr')

        im = shift(ref, [2, 0], mode='constant', order=0)
        imgIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.estimate(imgIn, ref).collect()[0][1]
        imgOut = reg.transform(imgIn, ref).first()[1]
        assert(allclose(ref[:-2, :], imgOut[:-2, :]))
        assert(allclose(paramOut, [2, 0]))

        im = shift(ref, [0, 2], mode='constant', order=0)
        imgIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.estimate(imgIn, ref).collect()[0][1]
        imgOut = reg.transform(imgIn, ref).first()[1]
        assert(allclose(ref[:, :-2], imgOut[:, :-2]))
        assert(allclose(paramOut, [0, 2]))

        im = shift(ref, [2, -2], mode='constant', order=0)
        imgIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.estimate(imgIn, ref).collect()[0][1]
        imgOut = reg.transform(imgIn, ref).first()[1]
        assert(allclose(ref[:-2, 2:], imgOut[:-2, 2:]))
        assert(allclose(paramOut, [2, -2]))

        im = shift(ref, [-2, 2], mode='constant', order=0)
        imgIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.estimate(imgIn, ref).collect()[0][1]
        imgOut = reg.transform(imgIn, ref).first()[1]
        assert(allclose(ref[2:, :-2], imgOut[2:, :-2]))
        assert(allclose(paramOut, [-2, 2]))

        # just that that applying a filter during registration runs
        # TODO add a check that shows this helps compensate for noisy pixels
        reg = Register('crosscorr').setFilter('median', 2)
        im = shift(ref, [-2, 2], mode='constant', order=0)
        imgIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.estimate(imgIn, ref).collect()[0][1]
        imgOut = reg.transform(imgIn, ref).first()[1]

    def test_crosscorr_volume(self):

        random.seed(42)
        ref = random.randn(25, 25, 3)

        im = shift(ref, [2, -2, 0], mode='constant', order=0)
        imgIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = Register('crosscorr').estimate(imgIn, ref).collect()[0][1]
        imgOut = Register('crosscorr').transform(imgIn, ref).first()[1]
        assert(allclose(paramOut, [[2, -2], [2, -2], [2, -2]]))
        assert(allclose(ref[:-2, 2:, :], imgOut[:-2, 2:, :]))

    def test_reference_2d(self):

        random.seed(42)
        im0 = random.randn(25, 25).astype('uint16')
        im1 = random.randn(25, 25).astype('uint16')
        im2 = random.randn(25, 25).astype('uint16')
        imgIn = ImagesLoader(self.sc).fromArrays([im0, im1, im2])
        ref = Register.reference(imgIn)
        assert(allclose(ref, (im0 + im1 + im2) / 3))

        print(imgIn.keys().collect())

        ref = Register.reference(imgIn, startIdx=0, stopIdx=2)
        assert(allclose(ref, (im0 + im1) / 2))

        ref = Register.reference(imgIn, startIdx=1, stopIdx=2)
        assert(allclose(ref, im1))

    def test_reference_3d(self):

        random.seed(42)
        im0 = random.randn(25, 25, 3).astype('uint16')
        im1 = random.randn(25, 25, 3).astype('uint16')
        imgIn = ImagesLoader(self.sc).fromArrays([im0, im1])
        ref = Register.reference(imgIn)
        assert(allclose(ref, (im0 + im1) / 2))
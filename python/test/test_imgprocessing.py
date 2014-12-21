import shutil
import tempfile
from test_utils import PySparkTestCase
from numpy import random, allclose, arange
from scipy.ndimage.interpolation import shift
from thunder.imgprocessing.register import Register
from thunder.rdds.fileio.imagesloader import ImagesLoader


class ImgprocessingTestCase(PySparkTestCase):
    def setUp(self):
        super(ImgprocessingTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(ImgprocessingTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestRegistrationBasic(ImgprocessingTestCase):

    def test_save_load(self):

        # test basic saving a loading functionality
        # new registration methods should add tests
        # for loading and saving

        random.seed(42)
        ref = random.randn(25, 25)

        im = shift(ref, [2, 0], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)

        reg = Register('crosscorr')
        model1 = reg.fit(imin, ref)

        t = tempfile.mkdtemp()
        model1.save(t + '/test.json')
        model2 = Register.load(t + '/test.json')

        out1 = model1.transform(imin).first()[1]
        out2 = model2.transform(imin).first()[1]

        assert(allclose(out1, out2))

    def test_run(self):

        # tests the run method which combines fit and transform

        random.seed(42)
        ref = random.randn(25, 25)

        im = shift(ref, [2, 0], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)

        reg = Register('crosscorr')
        model = reg.fit(imin, ref)
        out1 = model.transform(imin).first()[1]
        out2 = reg.run(imin, ref).first()[1]

        assert(allclose(out1, out2))

    def test_reference_2d(self):

        # test default reference calculation in 2D

        random.seed(42)
        im0 = random.randn(25, 25).astype('uint16')
        im1 = random.randn(25, 25).astype('uint16')
        im2 = random.randn(25, 25).astype('uint16')
        imin = ImagesLoader(self.sc).fromArrays([im0, im1, im2])

        ref = Register('crosscorr').reference(imin)
        assert(allclose(ref, (im0 + im1 + im2) / 3))

        print(imin.keys().collect())

        ref = Register('crosscorr').reference(imin, startidx=0, stopidx=2)
        assert(allclose(ref, (im0 + im1) / 2))

        ref = Register('crosscorr').reference(imin, startidx=1, stopidx=2)
        assert(allclose(ref, im1))

    def test_reference_3d(self):

        # test default reference calculation in 3D

        random.seed(42)
        im0 = random.randn(25, 25, 3).astype('uint16')
        im1 = random.randn(25, 25, 3).astype('uint16')
        imin = ImagesLoader(self.sc).fromArrays([im0, im1])
        ref = Register('crosscorr').reference(imin)
        assert(allclose(ref, (im0 + im1) / 2))


class TestCrossCorr(ImgprocessingTestCase):

    def test_crosscorr_image(self):

        random.seed(42)
        ref = random.randn(25, 25)

        reg = Register('crosscorr')

        im = shift(ref, [2, 0], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)
        paramout = reg.fit(imin, ref).transformations[0].delta
        imout = reg.run(imin, ref).first()[1]
        assert(allclose(ref[:-2, :], imout[:-2, :]))
        assert(allclose(paramout, [2, 0]))

        im = shift(ref, [0, 2], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)
        paramout = reg.fit(imin, ref).transformations[0].delta
        imout = reg.run(imin, ref).first()[1]
        assert(allclose(ref[:, :-2], imout[:, :-2]))
        assert(allclose(paramout, [0, 2]))

        im = shift(ref, [2, -2], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)
        paramout = reg.fit(imin, ref).transformations[0].delta
        imout = reg.run(imin, ref).first()[1]
        assert(allclose(ref[:-2, 2:], imout[:-2, 2:]))
        assert(allclose(paramout, [2, -2]))

        im = shift(ref, [-2, 2], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)
        paramout = reg.fit(imin, ref).transformations[0].delta
        imout = reg.run(imin, ref).first()[1]
        assert(allclose(ref[2:, :-2], imout[2:, :-2]))
        assert(allclose(paramout, [-2, 2]))

    def test_crosscorr_volume(self):

        random.seed(42)
        ref = random.randn(25, 25, 3)
        im = shift(ref, [2, -2, 0], mode='constant', order=0)
        imin = ImagesLoader(self.sc).fromArrays(im)

        # use 3D cross correlation
        paramout = Register('crosscorr').fit(imin, ref).transformations[0].delta
        imout = Register('crosscorr').run(imin, ref).first()[1]
        assert(allclose(paramout, [2, -2, 0]))
        assert(allclose(ref[:-2, 2:, :], imout[:-2, 2:, :]))

        # use 2D cross correlation on each plane
        paramout = Register('planarcrosscorr').fit(imin, ref).transformations[0].delta
        imout = Register('planarcrosscorr').run(imin, ref).first()[1]
        assert(allclose(paramout, [[2, -2], [2, -2], [2, -2]]))
        assert(allclose(ref[:-2, 2:, :], imout[:-2, 2:, :]))

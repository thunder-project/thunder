from numpy import arange, ndarray, argmax, unravel_index

from thunder.rdds.images import Images
from thunder.utils.common import checkParams


class Register(object):

    def __new__(cls, method="crosscorr"):

        checkParams(method, ["crosscorr"])

        if method == "crosscorr":
            return super(Register, cls).__new__(CrossCorr)
        else:
            raise Exception('Registration method not recognized')

    def getTransform(self, im, ref):
        raise NotImplementedError

    def applyTransform(self, im, transform):
        raise NotImplementedError

    def setFilter(self, filter='median', param=2):
        """
        Set a filter to apply to images before registration.

        The filtering will be applied to both the reference and
        image to compute the transformation parameters, but the filtering
        will not be applied to the images themselves.

        Parameters
        ----------

        filter : str, optional, default = 'median'
            Which filter to use (options are 'median' and 'gaussian')

        param : int, optional, default = 2
            Parameter to provide to filtering function (e.g. size for median filter)

        See also
        --------
        Images.medianFilter : apply median filter to images
        Images.gaussianFilter : apply gaussian filter to images

        """

        checkParams(filter, ['median', 'gaussian'])

        if filter == 'median':
            from scipy.ndimage.filters import median_filter
            self._filter = lambda x: median_filter(x, param)

        if filter == 'gaussian':
            from scipy.ndimage.filters import gaussian_filter
            self._filter = lambda x: gaussian_filter(x, param)

        return self

    def filter(self, im):
        """
        Apply filtering, and if not set, return image unchanged
        """

        if hasattr(self, '_filter'):
            return self._filter(im)
        else:
            return im

    @staticmethod
    def reference(images, method='mean', startIdx=None, stopIdx=None):
        """
        Compute a reference image for use in registration.

        Parameters
        ----------
        method : str, optional, default = 'mean'
            How to compute the reference

        startidx : int, optional, default = None
            Starting index if computing a mean over a specified range

        stopidx : int, optional, default = None
            Stopping index if computing a mean over a specified range

        """

        # TODO easy option for using the mean of the middle n images
        # TODO fix inclusive behavior to match e.g. image loading

        checkParams(method, ['mean'])

        if method == 'mean':
            if startIdx is not None and stopIdx is not None:
                idxRange = lambda x: startIdx <= x < stopIdx
                n = stopIdx - startIdx
                ref = images.filterOnKeys(idxRange)
            else:
                ref = images
                n = images.nimages
            refval = ref.sum() / (1.0 * n)
            return refval.astype(images.dtype)

    @staticmethod
    def _applyVol(vol, func):
        """
        Apply a function to an image, or a volume (plane-by-plane).
        """

        if vol.ndim == 2:
            return func(vol)
        else:
            vol.setflags(write=True)
            for z in arange(0, vol.shape[2]):
                vol[:, :, z] = func(vol[:, :, z])
            return vol

    @staticmethod
    def _checkReference(images, reference):
        """
        Check the dimensions and type of a reference (relative to an Images object),
        as well as check that the images / volumes themselves are either 2D or 3D
        """

        if isinstance(reference, ndarray):
            if reference.shape != images.dims.count:
                raise Exception('Dimensions of reference %s do not match dimensions of data %s' %
                                (reference.shape, images.dims.count))
            if len(images.dims.count) not in set([2, 3]):
                raise Exception('Number of image dimensions %s must be 2 or 3' % (len(images.dims.count)))
        else:
            raise Exception('Reference must be an array')

    def estimate(self, images, reference):
        """
        Estimate registration parameters on a collection of images / volumes.

        Will return a list of registration parameters to the driver, rather
        than the registered images / volumes themselves.

        Parameters
        ----------
        images : Images
            An Images object containing the images / volumes to estimate registration for

        reference : ndarray
            The reference image / volume to estimate registration against

        Returns
        -------
        params : list
            Registration parameters, one per image. Will be returned as a list of key-value pairs,
            where the key is the same key used to identify each image / volume in the Images object,
            and the value is a list of registration parameters (in whatever format provided by
            the registration function; e.g. for CrossCorr will return a list of deltas in x and y)

        See also
        --------
        Register.transform : apply transformations
        """

        if not (isinstance(images, Images)):
            raise Exception('Input data must be Images or a subclass')

        self._checkReference(images, reference)

        # apply filtering to reference if defined
        if hasattr(self, '_filter'):
            reference = self._applyVol(reference.copy(), self.filter)

        # broadcast the reference (a potentially very large array)
        bcReference = images.rdd.context.broadcast(reference)

        # estimate the transform parameters on an image / volume
        def params(im, ref):
            if im.ndim == 2:
                return self.getTransform(self.filter(im), ref.value)
            else:
                t = []
                for z in arange(0, im.shape[2]):
                    t.append(self.getTransform(self.filter(im[:, :, z]), ref.value[:, :, z]))
            return t

        from thunder import Series
        return Series(images.rdd.mapValues(lambda x: params(x, bcReference)))

    def transform(self, images, reference):
        """
        Apply registration to a collection of images / volumes.

        Parameters
        ----------
        images : Images
            An Images object containing the images / volumes to apply registration to

        reference : ndarray
            The reference image / volume to register against
        """

        if not (isinstance(images, Images)):
            raise Exception('Input data must be Images or a subclass')

        self._checkReference(images, reference)

        # apply filtering to reference if defined
        if hasattr(self, '_filter'):
            reference = self._applyVol(reference, self.filter)

        # broadcast the reference (a potentially very large array)
        referenceBC = images.rdd.context.broadcast(reference)

        # compute and apply transformation on an image / volume
        def register(im, ref):
            if im.ndim == 2:
                t = self.getTransform(self.filter(im), ref.value)
                return self.applyTransform(im, t)
            else:
                im.setflags(write=True)
                for z in arange(0, im.shape[2]):
                    t = self.getTransform(self.filter(im[:, :, z]), ref.value[:, :, z])
                    im[:, :, z] = self.applyTransform(im[:, :, z], t)
                return im

        # return the transformed volumes
        newRdd = images.rdd.mapValues(lambda x: register(x, referenceBC))
        return Images(newRdd).__finalize__(images)


class CrossCorr(Register):
    """
    Perform affine (translation) registration using cross-correlation
    """

    def getTransform(self, im, ref):

        from numpy.fft import fft2, ifft2

        fref = fft2(ref)
        fim = fft2(im)
        c = abs(ifft2((fim * fref.conjugate())))
        d0, d1 = unravel_index(argmax(c), c.shape)
        if d0 > im.shape[0] // 2:
            d0 -= im.shape[0]
        if d1 > im.shape[1] // 2:
            d1 -= im.shape[1]

        return [d0, d1]

    def applyTransform(self, im, transform):

        from scipy.ndimage.interpolation import shift
        return shift(im, map(lambda x: -x, transform), mode='nearest')
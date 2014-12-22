""" Transformations produced by registration methods """


class Transformation(object):
    """ Base class for transformations """

    def apply(self, im):
        raise NotImplementedError


class Displacement(Transformation):

    def __init__(self, delta=None):
        self.delta = delta

    def apply(self, im):
        """
        Apply an n-dimensional displacement by shifting an image or volume.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift

        return shift(im, map(lambda x: -x, self.delta), mode='nearest')

    def __repr__(self):
        return "Displacement(delta=%s)" % str(self.delta)


class PlanarDisplacement(Transformation):

    def __init__(self, delta=None):
        self.delta = delta

    def apply(self, im):
        """
        Apply an 2D displacement by shifting each plane of a volume.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift

        if im.ndim == 2:
            return shift(im,  map(lambda x: -x, self.delta[0]))
        else:
            im.setflags(write=True)
            for z in range(0, im.shape[2]):
                im[:, :, z] = shift(im[:, :, z],  map(lambda x: -x, self.delta[z]), mode='nearest')
            return im

    def __repr__(self):
        return "PlanarDisplacement(delta=%s)" % str(self.delta)

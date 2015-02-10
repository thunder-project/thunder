""" Transformations produced by registration methods """
from thunder.utils.decorators import serializable


class Transformation(object):
    """ Base class for transformations """

    def apply(self, im):
        raise NotImplementedError


@serializable
class Displacement(Transformation):
    """
    Class for transformations based on spatial displacements.

    Can be applied to either images or volumes.

    Parameters
    ----------
    delta : list
        A list of spatial displacements for each dimensino,
        e.g. [10,5,2] for a displacement of 10 in x, 5 in y, 2 in z
    """

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
        return "Displacement(delta=%s)" % repr(self.delta)


@serializable
class PlanarDisplacement(Transformation):
    """
    Class for transformations based on two-dimensional spatial displacements.

    Applied separately to each plane of a three-dimensional volume.

    Parameters
    ----------
    delta : list
        A nested list, where the first list is over planes, and
        for each plane a list of [x,y] displacements
    """

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
        return "PlanarDisplacement(delta=%s)" % repr(self.delta)

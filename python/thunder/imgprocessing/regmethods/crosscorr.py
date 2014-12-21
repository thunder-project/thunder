""" Registration methods based on cross correlation """

from thunder.imgprocessing.register import RegisterMethod
from thunder.imgprocessing.regmethods.utils import computeDisplacement


class CrossCorr(RegisterMethod):
    """
    Translation using cross correlation.
    """

    def getTransform(self, im, ref):
        """
        Compute displacement between an image or volume and reference.

        Displacements are computed using the dimensionality of the inputs,
        so will be 2D for images and 3D for volumes.

        Parameters
        ----------
        im : ndarray
            The image or volume

        ref : ndarray
            The reference image or volume

        """

        from thunder.imgprocessing.transformation import Displacement

        delta = computeDisplacement(im, ref)

        return Displacement(delta)


class PlanarCrossCorr(RegisterMethod):
    """
    Translation using cross correlation on each plane.
    """

    def getTransform(self, im, ref):
        """
        Compute the planar displacement between an image or volume and reference.

        For 3D data (volumes), this will compute a separate 2D displacement for each plane.
        For 2D data (images), this will compute the displacement for the single plane
        (and will be the same as using CrossCorr).

        Parameters
        ----------
        im : ndarray
            The image or volume

        ref : ndarray
            The reference image or volume
        """

        from thunder.imgprocessing.transformation import PlanarDisplacement

        delta = []

        if im.ndim == 2:
            delta.append(computeDisplacement(im, ref))
        else:
            for z in range(0, im.shape[2]):
                delta.append(computeDisplacement(im[:, :, z], ref[:, :, z]))

        return PlanarDisplacement(delta)
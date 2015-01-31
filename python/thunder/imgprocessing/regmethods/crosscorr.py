""" Registration methods based on cross correlation """

from numpy import ndarray

from thunder.rdds.images import Images
from thunder.imgprocessing.registration import RegistrationMethod
from thunder.imgprocessing.regmethods.utils import computeDisplacement, computeReferenceMean, checkReference


class CrossCorr(RegistrationMethod):
    """
    Translation using cross correlation.
    """

    def prepare(self, images, startidx=None, stopidx=None):
        """
        Prepare cross correlation by computing or specifying a reference image.

        Parameters
        ----------
        images : ndarray or Images object
            Images to compute reference from, or a single image to set as reference

        See computeReferenceMean.
        """

        if isinstance(images, Images):
            self.reference = computeReferenceMean(images, startidx, stopidx)
        elif isinstance(images, ndarray):
            self.reference = images
        else:
            raise Exception('Must provide either an Images object or a reference')

        return self

    def isPrepared(self, images):
        """
        Check if cross correlation is prepared by checking the dimensions of the reference.

        See checkReference.
        """

        if not hasattr(self, 'reference'):
            raise Exception('Reference not defined')
        else:
            checkReference(self.reference, images)

    def getTransform(self, im):
        """
        Compute displacement between an image or volume and reference.

        Displacements are computed using the dimensionality of the inputs,
        so will be 2D for images and 3D for volumes.

        Parameters
        ----------
        im : ndarray
            The image or volume
        """

        from thunder.imgprocessing.transformation import Displacement

        delta = computeDisplacement(im, self.reference)

        return Displacement(delta)


class PlanarCrossCorr(CrossCorr):
    """
    Translation using cross correlation on each plane.
    """

    def getTransform(self, im):
        """
        Compute the planar displacement between an image or volume and reference.

        Overrides method from CrossCorr.

        For 3D data (volumes), this will compute a separate 2D displacement for each plane.
        For 2D data (images), this will compute the displacement for the single plane
        (and will be the same as using CrossCorr).

        Parameters
        ----------
        im : ndarray
            The image or volume
        """

        from thunder.imgprocessing.transformation import PlanarDisplacement

        delta = []

        if im.ndim == 2:
            delta.append(computeDisplacement(im, self.reference))
        else:
            for z in range(0, im.shape[2]):
                delta.append(computeDisplacement(im[:, :, z], self.reference[:, :, z]))

        return PlanarDisplacement(delta)
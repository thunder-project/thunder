""" Shared utilities for registration methods """

from numpy import ndarray

from thunder.rdds.images import Images


def computeReferenceMean(images, startidx, stopidx):
    """
    Compute a reference by taking the mean across images.

    Parameters
    ----------
    images : Images
            An Images object containg the image / volumes to compute reference from

    startidx : int, optional, default = None
        Starting index if computing a mean over a specified range

    stopidx : int, optional, default = None
        Stopping index if computing a mean over a specified range

    Returns
    -------
    refval : ndarray
        The reference image / volume
    """

    if not (isinstance(images, Images)):
        raise Exception('Input data must be Images or a subclass')

    if startidx is not None and stopidx is not None:
        range = lambda x: startidx <= x < stopidx
        n = stopidx - startidx
        ref = images.filterOnKeys(range)
    else:
        ref = images
        n = images.nimages

    reference = (ref.sum() / float(n)).astype(images.dtype)

    return reference


def checkReference(images, reference):
    """
    Check that a reference is an ndarray and matches the dimensions of images.

    Parameters
    ----------
    images : Images
        An Images object containing the image / volumes to check against the reference

    reference : ndarray
        A reference image / volume
    """

    if isinstance(reference, ndarray):
        if reference.shape != images.dims.count:
            raise Exception('Dimensions of reference %s do not match dimensions of data %s' %
                            (reference.shape, images.dims.count))
        else:
            raise Exception('Reference must be an array')


def computeDisplacement(arry1, arry2):
    """
    Compute an optimal displacement between two ndarrays.

    Finds the displacement between two ndimensional arrays. Arrays must be
    of the same size. Algorithm uses a cross correlation, computed efficiently
    through an n-dimensional fft.

    Parameters
    ----------
    arry1 : ndarray
        The first array

    arry2 : ndarray
        The second array
    """

    from numpy.fft import fftn, ifftn
    from numpy import unravel_index, argmax

    # get fourier transforms
    f2 = fftn(arry2)
    f1 = fftn(arry1)

    # get cross correlation
    c = abs(ifftn((f1 * f2.conjugate())))

    # find location of maximum
    maxinds = unravel_index(argmax(c), c.shape)

    # fix displacements that are greater than half the total size
    pairs = zip(maxinds, arry1.shape)
    # cast to basic python int for serialization
    adjusted = [int(d - n) if d > n // 2 else int(d) for (d, n) in pairs]

    return adjusted
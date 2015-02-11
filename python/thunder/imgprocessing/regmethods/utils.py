""" Shared utilities for registration methods """

from numpy import ndarray

from thunder.rdds.images import Images


def computeReferenceMean(images, startIdx=None, stopIdx=None, defaultNImages=20):
    """
    Compute a reference by taking the mean across images.

    The default behavior is to take the mean across the center `defaultNImages` records
    in the Images RDD. If startIdx or stopIdx is specified, then the mean will be
    calculated across this range instead.

    Parameters
    ----------
    images : Images
            An Images object containing the image / volumes to compute reference from

    startIdx : int, optional, default = None
        Starting index if computing a mean over a specified range

    stopIdx : int, optional, default = None
        Stopping index (exclusive) if computing a mean over a specified range

    defaultNImages : int, optional, default = 20
        Number of images across which to calculate the mean if neither startIdx nor stopIdx
        is given.

    Returns
    -------
    refval : ndarray
        The reference image / volume
    """

    if not (isinstance(images, Images)):
        raise Exception('Input data must be Images or a subclass')

    doFilter = True
    if startIdx is None and stopIdx is None:
        n = images.nimages
        if n <= defaultNImages:
            doFilter = False
        else:
            ctrIdx = n / 2  # integer division
            halfWindow = defaultNImages / 2  # integer division
            parity = 1 if defaultNImages % 2 else 0
            startIdx = ctrIdx - halfWindow
            stopIdx = ctrIdx + halfWindow + parity
            n = stopIdx - startIdx
    else:
        if startIdx is None:
            startIdx = 0
        if stopIdx is None:
            stopIdx = images.nimages
        n = stopIdx - startIdx

    if doFilter:
        rangePredicate = lambda x: startIdx <= x < stopIdx
        ref = images.filterOnKeys(rangePredicate)
    else:
        ref = images

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
    maxInds = unravel_index(argmax(c), c.shape)

    # fix displacements that are greater than half the total size
    pairs = zip(maxInds, arry1.shape)
    adjusted = [d - n if d > n // 2 else d for (d, n) in pairs]

    return adjusted
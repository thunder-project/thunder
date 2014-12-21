""" Shared utilities for registration methods """


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
    adjusted = [d - n if d > n // 2 else d for (d, n) in pairs]

    return adjusted
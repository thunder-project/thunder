""" Common operations and utilities """


def pinv(mat):
    """ Compute pseudoinverse of a matrix """
    from scipy.linalg import inv
    from numpy import dot, transpose
    return dot(inv(dot(mat, transpose(mat))), mat)


def loadmatvar(filename, var):
    """ Load a variable from a MAT file"""
    from scipy.io import loadmat
    return loadmat(filename)[var]


def isrdd(data):
    """ Check whether data is an RDD or not"""
    dtype = type(data)
    import pyspark
    if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD):
        return True
    else:
        return False


def checkparams(param, opts):
    """ Check whether param is contained in opts (including lowercase), otherwise error"""
    if not param.lower() in opts:
        raise ValueError("Option must be one of %s, got %s" % (str(opts)[1:-1], param))


def smallest_float_type(dtype):
    """Returns the smallest floating point dtype to which the passed dtype can be safely cast.

    For integers and unsigned ints, this will generally be next floating point type larger than the integer type. So
    for instance, smallest_float_type('uint8') -> dtype('float16'), smallest_float_type('int16') -> dtype('float32'),
    smallest_float_type('uint32') -> dtype('float64').

    This function relies on numpy's promote_types function.
    """
    from numpy import dtype as dtype_func
    from numpy import promote_types
    intype = dtype_func(dtype)
    compsize = max(2, intype.itemsize)  # smallest float is at least 16 bits
    comptype = dtype_func('=f'+str(compsize))  # compare to a float of the same size
    return promote_types(intype, comptype)


def pil_to_array(pilImage):
    """
    Load a PIL image and return it as a numpy array.  Only supports greyscale images;
    the return value will be an M x N array.

    Adapted from matplotlib's pil_to_array, copyright 2009-2012 by John D Hunter.
    """
    def toarray(im_, dtype):
        """Return a 1D array of dtype."""
        from numpy import fromstring
        # Pillow wants us to use "tobytes"
        if hasattr(im_, 'tobytes'):
            x_str = im_.tobytes('raw', im_.mode)
        else:
            x_str = im_.tostring('raw', im_.mode)
        x_ = fromstring(x_str, dtype)
        return x_

    if pilImage.mode in ('RGBA', 'RGBX', 'RGB'):
        raise ValueError("Thunder only supports luminance / greyscale images; got image mode: '%s'" % pilImage.mode)
    if pilImage.mode == 'L':
        im = pilImage  # no need to luminance images
        # return MxN luminance array
        x = toarray(im, 'uint8')
        x.shape = im.size[1], im.size[0]
        return x
    elif pilImage.mode.startswith('I;16'):
        # return MxN luminance array of uint16
        im = pilImage
        if im.mode.endswith('B'):
            x = toarray(im, '>u2')
        else:
            x = toarray(im, '<u2')
        x.shape = im.size[1], im.size[0]
        return x.astype('=u2')
    elif pilImage.mode.startswith('I;32') or pilImage.mode == 'I':
        # default 'I' mode is 32 bit; see http://svn.effbot.org/public/tags/pil-1.1.7/libImaging/Unpack.c (at bottom)
        # return MxN luminance array of uint32
        im = pilImage
        if im.mode.endswith('B'):
            x = toarray(im, '>u4')
        else:
            x = toarray(im, '<u4')
        x.shape = im.size[1], im.size[0]
        return x.astype('=u4')
    elif pilImage.mode.startswith('F;16'):
        # return MxN luminance array of float16
        im = pilImage
        if im.mode.endswith('B'):
            x = toarray(im, '>f2')
        else:
            x = toarray(im, '<f2')
        x.shape = im.size[1], im.size[0]
        return x.astype('=f2')
    elif pilImage.mode.startswith('F;32') or pilImage.mode == 'F':
        # default 'F' mode is 32 bit; see http://svn.effbot.org/public/tags/pil-1.1.7/libImaging/Unpack.c (at bottom)
        # return MxN luminance array of float32
        im = pilImage
        if im.mode.endswith('B'):
            x = toarray(im, '>f4')
        else:
            x = toarray(im, '<f4')
        x.shape = im.size[1], im.size[0]
        return x.astype('=f4')
    else:  # try to convert to an rgba image
        raise ValueError("Thunder only supports luminance / greyscale images; got unknown image mode: '%s'"
                         % pilImage.mode)


def parseMemoryString(memstr):
    """Returns the size in bytes of memory represented by a Java-style 'memory string'

    parseMemoryString("150k") -> 150000
    parseMemoryString("2M") -> 2000000
    parseMemoryString("5G") -> 5000000000
    parseMemoryString("128") -> 128

    Recognized suffixes are k, m, and g. Parsing is case-insensitive.
    """
    if isinstance(memstr, basestring):
        import re
        regpat = r"""(\d+)([bBkKmMgG])?"""
        m = re.match(regpat, memstr)
        if not m:
            raise ValueError("Could not parse %s as memory specification; should be NUMBER[k|m|g]" % memstr)
        quant = int(m.group(1))
        units = m.group(2).lower()
        if units == "g":
            return int(quant * 1e9)
        elif units == 'm':
            return int(quant * 1e6)
        elif units == 'k':
            return int(quant * 1e3)
        return quant
    else:
        return int(memstr)


def smallestFloatType(dtype):
    """
    Returns the smallest floating point dtype to which the passed dtype can be safely cast.

    For integers and unsigned ints, this will generally be next floating point type larger than the integer type. So
    for instance, smallest_float_type('uint8') -> dtype('float16'), smallest_float_type('int16') -> dtype('float32'),
    smallest_float_type('uint32') -> dtype('float64').

    This function relies on numpy's promote_types function.
    """
    from numpy import dtype as dtypeFunc
    from numpy import promote_types
    inType = dtypeFunc(dtype)
    compSize = max(2, inType.itemsize)  # smallest float is at least 16 bits
    compType = dtypeFunc('=f'+str(compSize))  # compare to a float of the same size
    return promote_types(inType, compType)


def pil_to_array(pilImage):
    """
    Load a PIL image and return it as a numpy array.  Only supports greyscale images;
    the return value will be an M x N array.

    Adapted from matplotlib's pil_to_array, copyright 2009-2012 by John D Hunter
    """
    # This is intended to be used only with older versions of PIL, for which the new-style
    # way of getting a numpy array (np.array(pilimg)) does not appear to work in all cases.
    # np.array(pilimg) appears to work with Pillow 2.3.0; with PIL 1.1.7 it leads to
    # errors similar to the following:
    # In [15]: data = tsc.loadImages('/path/to/tifs/', inputformat='tif-stack')
    # In [16]: data.first()[1].shape
    # Out[16]: (1, 1, 1)
    # In [17]: data.first()[1]
    # Out[17]: array([[[ <PIL.TiffImagePlugin.TiffImageFile image mode=I;16 size=512x512 at 0x3B02B00>]]],
    # dtype=object)
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
        raise ValueError("Thunder only supports luminance / greyscale images in pil_to_array; got image mode: '%s'" %
                         pilImage.mode)
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
        raise ValueError("Thunder only supports luminance / greyscale images in pil_to_array; got unknown image " +
                         "mode: '%s'" % pilImage.mode)

def parseMemoryString(memStr):
    """
    Returns the size in bytes of memory represented by a Java-style 'memory string'

    parseMemoryString("150k") -> 150000
    parseMemoryString("2M") -> 2000000
    parseMemoryString("5G") -> 5000000000
    parseMemoryString("128") -> 128

    Recognized suffixes are k, m, and g. Parsing is case-insensitive.
    """
    if isinstance(memStr, basestring):
        import re
        regPat = r"""(\d+)([bBkKmMgG])?"""
        m = re.match(regPat, memStr)
        if not m:
            raise ValueError("Could not parse '%s' as memory specification; should be NUMBER[k|m|g]" % memStr)
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
        return int(memStr)


def raiseErrorIfPathExists(path, credentials=None):
    """
    The ValueError message will suggest calling with overwrite=True; this function is expected to be
    called from the various output methods that accept an 'overwrite' keyword argument.
    """
    # check that specified output path does not already exist
    from thunder.data.fileio.readers import getFileReaderForPath
    reader = getFileReaderForPath(path)(credentials=credentials)
    existing = reader.list(path, includeDirectories=True)
    if existing:
        raise ValueError("Path %s appears to already exist. Specify a new directory, or call " % path +
                         "with overwrite=True to overwrite.")

def S3ConnectionWithAnon(credentials, anon=True):
    """
    Connect to S3 with automatic handling for anonymous access

    Parameters
    ----------
    credentials : dict
        AWS access key ('access') and secret access key ('secret')

    anon : boolean, optional, default = True
        Whether to make an anonymous connection if credentials fail to authenticate
    """
    from boto.s3.connection import S3Connection
    from boto.exception import NoAuthHandlerFound

    try:
        conn = S3Connection(aws_access_key_id=credentials['access'],
                            aws_secret_access_key=credentials['secret'])
        return conn

    except NoAuthHandlerFound:
        if anon:
            conn = S3Connection(anon=True)
            return conn
        else:
            raise
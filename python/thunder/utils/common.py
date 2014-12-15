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


def selectByMatchingPrefix(param, opts):
    """Given a string parameter and a sequence of possible options, returns an option that is uniquely
    specified by matching its prefix to the passed parameter.

    The match is checked without sensitivity to case.

    Throws IndexError if none of opts starts with param, or if multiple opts start with param.

    >> selectByMatchingPrefix("a", ["aardvark", "mongoose"])
    "aardvark"
    """
    lparam = param.lower()
    hits = [opt for opt in opts if opt.lower().startswith(lparam)]
    nhits = len(hits)
    if nhits == 1:
        return hits[0]
    if nhits:
        raise IndexError("Multiple matches for for prefix '%s': %s") % (param, hits)
    else:
        raise IndexError("No matches for prefix '%s' found in options %s") % (param, opts)


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


""" Common operations and utilities """


def pinv(mat):
    """ Compute pseudoinverse of a matrix """
    from scipy.linalg import inv
    from numpy import dot, transpose
    return dot(inv(dot(mat, transpose(mat))), mat)


def loadMatVar(filename, var):
    """ Load a variable from a MAT file"""
    from scipy.io import loadmat
    return loadmat(filename)[var]


def isRdd(data):
    """ Check whether data is an RDD or not"""
    dtype = type(data)
    import pyspark
    if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD):
        return True
    else:
        return False


def checkParams(param, opts):
    """ Check whether param is contained in opts (including lowercase), otherwise error"""
    if not param.lower() in opts:
        raise ValueError("Option must be one of %s, got %s" % (str(opts)[1:-1], param))


def smallestFloatType(dtype):
    """Returns the smallest floating point dtype to which the passed dtype can be safely cast.

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


def parseMemoryString(memStr):
    """Returns the size in bytes of memory represented by a Java-style 'memory string'

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


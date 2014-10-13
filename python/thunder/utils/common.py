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
def isrdd(data):
    """ Check whether data is an RDD or not"""
    dtype = type(data)
    import pyspark
    if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD):
        return True
    else:
        return False


def checkparams(param, opts):
    """ Check whether param is contained in opts (including lowercase version),
    return error otherwise
    """
    if not param.lower() in opts:
        raise ValueError("Option must be one of %s, got %s" % (str(opts)[1:-1], param))

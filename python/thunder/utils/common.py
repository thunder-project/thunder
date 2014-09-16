
import pyspark


def isrdd(data):
    """ Check whether data is an RDD or not"""

    dtype = type(data)
    if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD):
        return True
    else:
        return False
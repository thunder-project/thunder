"""
Utilities for loading and preprocessing data
"""

import pyspark

from numpy import array, mean, cumprod, append, mod, ceil, size
from scipy.signal import butter, lfilter


class Dimensions(object):

    def __init__(self, distinctvals, rng):
        self.min = map(lambda i: min(distinctvals[i]), rng)
        self.max = map(lambda i: max(distinctvals[i]), rng)
        self.num = map(lambda i: size(distinctvals[i]), rng)


class DataLoader(object):
    """Class for loading lines of a data file"""

    def __init__(self, nkeys):
        def func(line):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys:])
            keys = tuple(int(x) for x in vec[:nkeys])
            return keys, ts

        self.func = func

    def get(self, y):
        return self.func(y)


class DataPreProcessor(object):
    """Class for preprocessing data"""

    def __init__(self, preprocessmethod):
        if preprocessmethod == "sub":
            func = lambda y: y - mean(y)

        if preprocessmethod == "dff":
            def func(y):
                mnval = mean(y)
                return (y - mnval) / (mnval + 0.1)

        if preprocessmethod == "raw":
            func = lambda x: x

        if preprocessmethod == "dff-highpass":
            fs = 1
            nyq = 0.5 * fs
            cutoff = (1.0/360) / nyq
            b, a = butter(6, cutoff, "highpass")

            def func(y):
                mnval = mean(y)
                y = (y - mnval) / (mnval + 0.1)
                return lfilter(b, a, y)

        self.func = func

    def get(self, y):
        return self.func(y)


def getdims(data):
    """Get min, max, and number of unique keys along
    each dimension

    :param data: RDD of data points as key value pairs
    :return dims: Dimensions of the data (max, min, and num)
    """
    dtype = type(data)
    if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD):
        entry = data.first()[0]
        rng = range(0, size(entry))
        if size(entry) == 1:
            distinctvals = array([data.map(lambda (k, _): k).distinct().collect()])
        else:
            distinctvals = map(lambda i: data.map(lambda (k, _): k[i]).distinct().collect(), rng)

        return Dimensions(distinctvals, rng)
    else:
        entry = data[0][0]
        rng = range(0, size(entry))
        if size(entry) == 1:
            distinctvals = list(set(map(lambda x: x[0][0], data)))
        else:
            distinctvals = map(lambda i: list(set(map(lambda x: x[0][i], data))), rng)
        return Dimensions(distinctvals, rng)


def subtoind(data, dims):
    """Convert subscript indexing to linear indexing

    :param data: RDD with subscript indices as keys
    :param dims: Array with maximum along each dimension
    :return RDD with linear indices as keys
    """
    if size(dims) > 1:
        dimprod = cumprod(dims)[0:-1]
        return data.map(lambda (k, v):
                        (sum(map(lambda (x, y): (x - 1) * y, zip(k[1:], dimprod))) + k[0], v))
    else:
        return data


def indtosub(data, dims):
    """Convert linear indexing to subscript indexing

    :param data: RDD with linear indices as keys
    :param dims: Array with maximum along each dimension
    :return RDD with sub indices as keys
    """
    if size(dims) > 1:
        dimprod = zip(dims, append(1, cumprod(dims)[0:-1]))
        return data.map(lambda (k, v):
                        (tuple(map(lambda (x, y): int(mod(ceil(float(k)/y) - 1, x) + 1), dimprod)), v))
    else:
        return data


def load(sc, datafile, preprocessmethod="raw", nkeys=3):
    """Load data from a text file with format
    <k1> <k2> ... <t1> <t2> ...
    where <k1> <k2> ... are keys (Int) and <t1> <t2> ... are the data values (Double)
    If multiple keys are provided (e.g. x, y, z), they are converted to linear indexing

    :param sc: SparkContext
    :param datafile: Location of raw data
    :param preprocessmethod: Type of preprocessing to perform ("raw", "dff", "sub")
    :param nkeys: Number of keys per data point
    :return data: RDD of data points as key value pairs
    """

    lines = sc.textFile(datafile)
    loader = DataLoader(nkeys)

    data = lines.map(loader.get)

    if preprocessmethod != "raw":
        preprocessor = DataPreProcessor(preprocessmethod)
        data = data.mapValues(preprocessor.get)

    return data



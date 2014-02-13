"""
Utilities for loading and preprocessing data
"""

from numpy import array, mean, cumprod, append, mod, ceil, size


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

        self.func = func

    def get(self, y):
        return self.func(y)


def getdims(data):
    """Get maximum dimensions of data based on keys

    :param data: RDD of data points as key value pairs
    :return dims: Dimensions of the data
    """
    entry = data.first()[0]
    if size(entry) == 1:
        dims = array([data.map(lambda (k, _): k).reduce(max)])
    else:
        dims = map(lambda i: data.map(lambda (k, _): k[i]).reduce(max), range(0, size(entry)))
    return dims


def subtoind(data, dims):
    """Convert subscript indexing to linear indexing

    :param data: RDD with subscript indices as keys
    :param dims: dimensions
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
    :param dims: dimensions
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



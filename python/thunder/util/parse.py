"""
utilities for loading and parsing data
"""

from numpy import array, mean


class Parser(object):
    """class for parsing lines of a data file"""

    def __init__(self, nkeys, keepkeys, preprocessor):
        if keepkeys == "false":
            def func(line):
                vec = [float(x) for x in line.split(' ')]
                ts = preprocessor.get(array(vec[nkeys:]))
                return ts

        if (keepkeys == "true") | (keepkeys == "linear"):
            def func(line):
                vec = [float(x) for x in line.split(' ')]
                ts = preprocessor.get(array(vec[nkeys:]))
                keys = tuple(int(x) for x in vec[:nkeys])
                return keys, ts

        self.func = func

    def get(self, y):
        return self.func(y)


class Preprocessor(object):
    """class for preprocessing data points"""

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


def parse(lines, preprocessmethod="raw", nkeys=3, keepkeys="false"):
    """function for parsing input data
    assumes data are lines of a text file (separated by spaces)
    and the first "nkeys" entries are the keys (optional))

    arguments:
    data - RDD of raw data points (lines of text, numbers separated by spaces)
    preprocessmethod - how to preprocess the data ("raw", "dff", "sub")
    nkeys - number of keys (int)
    keepkeys - whether to keep the keys and in what form ("true", "false", "linear")

    TODO: add a loader for small helper matrices, text or matlab format
    """

    preprocessor = Preprocessor(preprocessmethod)
    parser = Parser(nkeys, keepkeys, preprocessor)
    data = lines.map(lambda x: parser.get(x))

    if keepkeys == "linear":
        if nkeys == 2:
            k0 = data.map(lambda (k, _): k[0])
            mx_k0 = k0.reduce(max)
            data = data.map(lambda (k, v): (k[0] + (k[1] - 1) * mx_k0 - 1, v))
        if nkeys == 3:
            k0 = data.map(lambda (k, _): k[0])
            k1 = data.map(lambda (k, _): k[1])
            mx_k0 = k0.reduce(max)
            mx_k1 = k1.reduce(max)
            data = data.map(lambda (k, v): (k[0] + (k[1] - 1) * mx_k0 + (k[2] - 1) * mx_k1 - 1, v))

    return data



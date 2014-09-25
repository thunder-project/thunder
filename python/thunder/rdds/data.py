from numpy import int16

FORMATS = {
    'int16': int16,
    'float': float
}


def parseMemoryString(memstr):
    """Returns the size in bytes of memory represented by a Java-style 'memory string'

    parseMemoryString("150k") -> 150000
    parseMemoryString("2M") -> 2000000
    parseMemoryString("5G") -> 5000000000
    parseMemoryString("128") -> 128

    Recognized suffixes are k, m, and g. Parsing is case-insensitive.
    """
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


class Data(object):

    def __init__(self, rdd):
        self.rdd = rdd

    def first(self):
        return self.rdd.first()

    def collect(self):
        return self.rdd.collect()

    def count(self):
        return self.rdd.count()

    def mean(self):
        return self.rdd.values().mean()

    def sum(self):
        return self.rdd.values().sum()

    def variance(self):
        return self.rdd.values().variance()

    def stdev(self):
        return self.rdd.values().stdev()

    def stats(self):
        return self.rdd.values().stats()

    def cache(self):
        self.rdd.cache()

from numpy import int16


class Data(object):
    """
    Generic base class for data types in thunder.

    All data types are backed by an RDD of key-value pairs
    where the key is a tuple identifier and the value is an array

    This base class mainly provides convienience functions for accessing
    properties of the object using the appropriate RDD methods
    """

    _metadata = []

    def __init__(self, rdd):
        self.rdd = rdd

    def __finalize__(self, other):
        """
        Lazily propagate attributes from other to self, only if attributes
        are not already defined in self

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate

        """
        if isinstance(other, Data):
            for name in self._metadata:
                if (getattr(other, name, None) is not None) and (getattr(self, name, None) is None):
                    object.__setattr__(self, name, getattr(other, name, None))
        return self

    @property
    def _constructor(self):
        raise NotImplementedError

    def first(self):
        return self.rdd.first()

    def take(self, *args, **kwargs):
        return self.rdd.take(*args, **kwargs)

    def values(self):
        return self.rdd.values()

    def keys(self):
        return self.rdd.keys()

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

    def filterOnKeys(self, func):
        return self._constructor(self.rdd.filter(lambda (k, v): func(k))).__finalize__(self)

    def filterOnValues(self, func):
        return self._constructor(self.rdd.filter(lambda (k, v): func(v))).__finalize__(self)


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


FORMATS = {
    'int16': int16,
    'float': float
}

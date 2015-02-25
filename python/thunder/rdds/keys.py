"""Helper functions and classes for working with keys"""

from numpy import mod, ceil, cumprod, append, size, inf, subtract


class Dimensions(object):
    """ Class for estimating and storing dimensions of data based on the keys """

    def __init__(self, values=[], n=3):
        self.min = tuple(map(lambda i: inf, range(0, n)))
        self.max = tuple(map(lambda i: -inf, range(0, n)))

        for v in values:
            self.merge(v)

    def merge(self, value):
        self.min = tuple(map(min, self.min, value))
        self.max = tuple(map(max, self.max, value))
        return self

    def mergeDims(self, other):
        self.min = tuple(map(min, self.min, other.min))
        self.max = tuple(map(max, self.max, other.max))
        return self

    @property
    def count(self):
        return tuple(map(lambda x: x + 1, map(subtract, self.max, self.min)))

    @classmethod
    def fromTuple(cls, tup):
        """ Generates a Dimensions object from the passed tuple. """
        mx = [v-1 for v in tup]
        mn = [0] * len(tup)
        return cls(values=[mx, mn], n=len(tup))

    def __str__(self):
        return "min=%s, max=%s, count=%s" % (str(self.min), str(self.max), str(self.count))

    def __repr__(self):
        return "Dimensions(values=[%s, %s], n=%d)" % (str(self.min), str(self.max), len(self.min))

    def __len__(self):
        return len(self.min)

    def __iter__(self):
        return iter(self.count)

    def __getitem__(self, item):
        return self.count[item]


def _indToSubConverter(dims, order='F', isOneBased=True):
    """
    Converter for changing linear indexing to subscript indexing

    See also
    --------
    Series.indtosub
    """
    _checkOrder(order)

    def indToSub_InlineOneBased(k, dimProd_):
        return tuple(map(lambda (x, y): int(mod(ceil(float(k)/y) - 1, x) + 1), dimProd_))

    def indToSub_InlineZeroBased(k, dimProd_):
        return tuple(map(lambda (x, y): int(mod(ceil(float(k+1)/y) - 1, x)), dimProd_))

    inlineFcn = indToSub_InlineOneBased if isOneBased else indToSub_InlineZeroBased

    if size(dims) > 1:
        if order == 'F':
            dimProd = zip(dims, append(1, cumprod(dims)[0:-1]))
        else:
            dimProd = zip(dims, append(1, cumprod(dims[::-1])[0:-1])[::-1])
        converter = lambda k: inlineFcn(k, dimProd)
    else:
        converter = lambda k: (k,)

    return converter


def _subToIndConverter(dims, order='F', isOneBased=True):
    """
    Converter for changing subscript indexing to linear indexing

    See also
    --------
    Series.subtoind
    """
    _checkOrder(order)

    def onebasedProd(xy):
        x, y = xy
        return (x - 1) * y

    def zerobasedProd(xy):
        x, y = xy
        return x * y

    def subToInd_InlineColmajor(k, dimProd_, prodFunc):
        return sum(map(prodFunc, zip(k[1:], dimProd_))) + k[0]

    def subToInd_InlineRowmajor(k, revDimProd, prodFunc):
        return sum(map(prodFunc, zip(k[:-1], revDimProd))) + k[-1]

    if size(dims) > 1:
        if order == 'F':
            dimProd = cumprod(dims)[0:-1]
            inlineFcn = subToInd_InlineColmajor
        else:
            dimProd = cumprod(dims[::-1])[0:-1][::-1]
            inlineFcn = subToInd_InlineRowmajor
        prodFcn = onebasedProd if isOneBased else zerobasedProd
        converter = lambda k: inlineFcn(k, dimProd, prodFcn)
    else:
        converter = lambda k: k[0]

    return converter


def _checkOrder(order):
    if order not in ('C', 'F'):
        raise TypeError("Order %s not understood, should be 'C' or 'F'.")


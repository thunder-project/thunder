"""
Utilities for loading and preprocessing data
"""

from numpy import array, mean, cumprod, append, mod, ceil, size, \
    polyfit, polyval, arange, percentile, inf, subtract, \
    asarray, ravel_multi_index
from scipy.signal import butter, lfilter
from thunder.utils.common import isrdd


# to Series
class Dimensions(object):
    """Helper class for estimating and storing dimensions of data
    based on the keys"""

    def __init__(self, values=[], n=3):
        self.min = tuple(map(lambda i: inf, range(0, n)))
        self.max = tuple(map(lambda i: -inf, range(0, n)))

        for v in values:
            self.merge(v)

    def merge(self, value):
        self.min = tuple(map(min, self.min, value))
        self.max = tuple(map(max, self.max, value))
        return self

    def count(self):
        return tuple(map(lambda x: x + 1, map(subtract, self.max, self.min)))

    def mergedims(self, other):
        self.min = tuple(map(min, self.min, other.min))
        self.max = tuple(map(max, self.max, other.max))
        return self
    

# to series
def getdims(data):
    """Get dimensions of data via the keys. Ranges can have arbtirary minima
    and maximum, but they must be contiguous (e.g. the indices of a dense matrix)."""

    def redfunc(left, right):
        return left.mergedims(right)

    if isrdd(data):
        entry = data.first()[0]
        n = size(entry)
        d = data.map(lambda (k, _): k).mapPartitions(lambda i: [Dimensions(i, n)]).reduce(redfunc)
    else:
        entry = data[0][0]
        rng = range(0, size(entry))
        d = Dimensions()
        if size(entry) == 1:
            distinctvals = list(set(map(lambda x: x[0][0], data)))
        else:
            distinctvals = map(lambda i: list(set(map(lambda x: x[0][i], data))), rng)
        d.max = tuple(map(max, distinctvals))
        d.min = tuple(map(min, distinctvals))

    return d

# to series
def _check_order(order):
    if not order in ('C', 'F'):
        raise TypeError("Order %s not understood, should be 'C' or 'F'.")

# to series
def subtoind(data, dims, order='F', onebased=True):
    """Convert subscript indexing to linear indexing

    Parameters
    ----------
    order : str, 'C' or 'F', default = 'F'
        Specifies row-major or column-major array indexing. See numpy.ravel_multi_index.

    onebased : boolean, default = True
        True if subscript indices start at 1, False if they start at 0
    """
    _check_order(order)

    def onebased_prod(x_y):
        x, y = x_y
        return (x - 1) * y

    def zerobased_prod(x_y):
        x, y = x_y
        return x * y

    def subtoind_inline_colmajor(k, dimprod, p_func):
        return sum(map(p_func, zip(k[1:], dimprod))) + k[0]

    def subtoind_inline_rowmajor(k, revdimprod, p_func):
        return sum(map(p_func, zip(k[:-1], revdimprod))) + k[-1]

    if size(dims) > 1:
        if order == 'F':
            dimprod = cumprod(dims)[0:-1]
            inline_fcn = subtoind_inline_colmajor
        else:
            dimprod = cumprod(dims[::-1])[0:-1][::-1]
            inline_fcn = subtoind_inline_rowmajor

        prod_fcn = onebased_prod if onebased else zerobased_prod

        if isrdd(data):
            return data.map(lambda (k, v): (inline_fcn(k, dimprod, prod_fcn), v))
        else:
            return map(lambda (k, v): (inline_fcn(k, dimprod, prod_fcn), v), data)

    else:
        if isrdd(data):
            return data.map(lambda (k, v): (k[0], v))
        else:
            return map(lambda (k, v): (k[0], v), data)

# to series
def indtosub(data, dims, order='F', onebased=True):
    """Convert linear indexing to subscript indexing

    Parameters
    ----------
    order : str, 'C' or 'F', default = 'F'
        Specifies row-major or column-major array indexing. See numpy.unravel_index.

    onebased : boolean, default = True
        True if generated subscript indices are to start at 1, False to start at 0
    """
    _check_order(order)

    def indtosub_inline_onebased(k, dimprod):
        return tuple(map(lambda (x, y): int(mod(ceil(float(k)/y) - 1, x) + 1), dimprod))

    def indtosub_inline_zerobased(k, dimprod):
        return tuple(map(lambda (x, y): int(mod(ceil(float(k+1)/y) - 1, x)), dimprod))

    inline_fcn = indtosub_inline_onebased if onebased else indtosub_inline_zerobased

    if size(dims) > 1:
        if order == 'F':
            dimprod = zip(dims, append(1, cumprod(dims)[0:-1]))
        else:
            dimprod = zip(dims, append(1, cumprod(dims[::-1])[0:-1])[::-1])

        if isrdd(data):
            return data.map(lambda (k, v): (inline_fcn(k, dimprod), v))
        else:
            return map(lambda (k, v): (inline_fcn(k, dimprod), v), data)
    else:
        return data

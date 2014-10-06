"""Helper functions and classes for working with keys"""

from numpy import mod, ceil, cumprod, append, size, inf, subtract


class Dimensions(object):
    """Class for estimating and storing dimensions of data
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

    def mergedims(self, other):
        self.min = tuple(map(min, self.min, other.min))
        self.max = tuple(map(max, self.max, other.max))
        return self

    @property
    def count(self):
        return tuple(map(lambda x: x + 1, map(subtract, self.max, self.min)))


def _indtosub_converter(dims, order='F', onebased=True):
    """Converter for changing linear indexing to subscript indexing

    See also
    --------
    Series.indtosub
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
        converter = lambda k: inline_fcn(k, dimprod)
    else:
        converter = lambda k: k

    return converter


def _subtoind_converter(dims, order='F', onebased=True):
    """Converter for changing subscript indexing to linear indexing

    See also
    --------
    Series.subtoind
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
        converter = lambda k: inline_fcn(k, dimprod, prod_fcn)
    else:
        converter = lambda k: k[0]

    return converter


def _check_order(order):
    if not order in ('C', 'F'):
        raise TypeError("Order %s not understood, should be 'C' or 'F'.")


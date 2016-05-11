from numpy import arange, r_, empty, zeros, random, where, prod, array
from itertools import product
from bolt.utils import allstack

from ..base import Base
import logging


class LocalChunks:
    """
    Simple helper class for a local chunked ndarray.
    """

    def __init__(self, values, shape, plan, dtype=None, padding=None):
        """
        Create a chunked ndarray.

        Parameters
        ----------
        values : ndarray (dtype 'object') or ndarrays
            Array containing all the chunks.

        shape : tuple
            Shape of the full unchunked array.

        plan : tuple
            Shape of each chunk (excluding edge effects).

        dtype : numpy dtype, optional, default = None
            dtype of chunks. If not given, will be inferred from first chunk.
        """
        self.values = values
        self.shape = shape
        self.plan = plan
        self.padding = padding
        if dtype is None:
            self.dtype = self.first.dtype
        if dtype is None:
            print("NONE!")

    @property
    def first(self):
        """
        First chunk
        """
        return self.values[tuple(zeros(len(self.values.shape)))]

    @staticmethod
    def chunk(arr, plan, padding=None):
        """
        Created a chunked array from a full array and a chunking plan.

        Parameters
        ----------
        array : ndarray
            Array that will be broken into chunks

        plan : tuple
            Shape of desired chunks (up to edge effects)

        padding : tuple or int
            Amount of padding along each dimensions for chunks. If an int, then
            the same amount of padding is used for all dimensions

        Returns
        -------
        LocalChunks
        """

        if padding is None:
            pad = arr.ndim*(0,)
        elif isinstance(padding, int):
            pad = arr.ndim*(padding,)
        else:
            pad = padding

        shape = arr.shape

        if any([x + y > z for x, y, z in zip(plan, pad, shape)]):
            raise ValueError("Chunk sizes %s plus padding sizes %s cannot exceed value dimensions %s along any axis"
                             % (tuple(plan), tuple(pad), tuple(shape)))

        if any([x > y for x, y in zip(pad, plan)]):
            raise ValueError("Padding sizes %s cannot exceed chunk sizes %s along any axis"
                             % (tuple(pad), tuple(plan)))

        breaks = [r_[arange(0, n, s), n] for n, s in zip(shape, plan)]
        limits = [zip(array(b[:-1])-p, array(b[1:])+p) for b, p in zip(breaks, pad)]
        slices = product(*[[slice(x[0], x[1]) for x in l] for l in limits])
        vals = [arr[s] for s in slices]
        newarr = empty(len(vals), dtype=object)
        for i in range(len(vals)):
            newarr[i] = vals[i]
        newsize = [b.shape[0]-1 for b in breaks]
        newarr = newarr.reshape(*newsize)
        return LocalChunks(newarr, shape, plan, dtype=arr.dtype, padding=padding)

    def unchunk(self):
        """
        Reconstitute the chunked array back into a full ndarray.

        Returns
        -------
        ndarray
        """
        if self.padding is not None:
            shape = self.values.shape
            arr = empty(shape, dtype=object)
            for inds in product([arange(s) for s in shape]):
                slices = []
                for i, p, n in zip(inds, self.padding, shape):
                    start = None if i == 0 else p
                    stop = None if i == n else -p
                    slices += slice(start, stop, None)
                arr[inds] = arr[inds][slices]
        else:
            arr = self.values

        return allstack(self.values.tolist())

    def map(self, func, value_shape=None, dtype=None):

        if value_shape is None or dtype is None:
            # try to compute the size of each mapped element by applying func to a random array
            try:
                mapped = func(random.randn(*self.plan).astype(self.dtype))
            # if this fails, try to use the first block instead
            except Exception:
                mapped = func(self.first)
            if value_shape is None:
                value_shape = mapped.shape
            if dtype is None:
                dtype = mapped.dtype

        blocked_dims = where(array(self.plan) != array(self.shape))[0]
        unblocked_dims = where(array(self.plan) == array(self.shape))[0]

        # check that no dimensions are dropped
        if len(value_shape) != len(self.plan):
            raise NotImplementedError('map on chunked array cannot drop dimensions')

        # check that chunked dimensions did not change shape
        if any([value_shape[i] != self.plan[i] for i in blocked_dims]):
            raise ValueError('map cannot change the sizes of chunked dimensions')

        newshape = [value_shape[i] if i in unblocked_dims else self.shape[i] for i in range(self.values.ndim)]
        newshape = tuple(newshape)

        c = prod(self.values.shape)
        mapped = empty(c, dtype=object)
        for i, a in enumerate(self.values.flatten()):
            mapped[i] = func(a)
        return LocalChunks(mapped.reshape(self.values.shape), newshape, value_shape, dtype)

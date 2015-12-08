from numpy import asarray, maximum, minimum, add, ndarray, prod, ufunc, array, mean, std, size
from bolt.utils import inshape, tupleize
from bolt.base import BoltArray

class Data(object):
    """
    Generic base class for data types.

    All data types are backed by a bolt array.

    This base class mainly provides convenience functions for accessing
    properties of the object using methods appropriate for the
    underlying computational backend.
    """
    _metadata = ['dtype', 'shape', 'mode']

    def __init__(self, values, mode='local'):
        self._values = values
        if isinstance(values, BoltArray):
            mode = 'spark'
        if isinstance(values, ndarray):
            mode = 'local'
        self._mode = mode

    def __repr__(self):
        s = self.__class__.__name__
        s += '\n%s: %s' % ('mode', getattr(self, 'mode'))
        for k in self._metadata:
            v = getattr(self, k)
            output = str(v)
            if len(output) > 50:
                output = output[0:50].strip() + ' ... '
                if output.lstrip().startswith('['):
                    output += '] '
                if hasattr(v, '__len__'):
                    output += '(length: %d)' % len(v)
            if not k == 'mode':
                s += '\n%s: %s' % (k, output)
        return s

    def __finalize__(self, other, noprop=()):
        """
        Lazily propagate attributes from other to self, only if attributes
        are not already defined in self

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate

        noprop : iterable of string attribute names, default empty tuple
            Attributes found in noprop will *not* have their values propagated forward.
        """
        if isinstance(other, Data):
            for name in self._metadata:
                if name not in noprop:
                    attr = getattr(other, name, None)
                    if (attr is not None) and (getattr(self, name, None) is None):
                        object.__setattr__(self, name, attr)
        return self

    def __getitem__(self, item):
        return self._values.__getitem__(item)

    @property
    def _constructor(self):
        return Data

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def shape(self):
        return self._values.shape

    @property
    def mode(self):
        return self._mode

    @property
    def values(self):
        return self._values

    def tospark(self):
        pass

    def tolocal(self):
        return

    def toarray(self):
        """
        Return all records to the driver as a numpy array

        This will be slow for large datasets, and may exhaust the available memory on the driver.
        """
        return asarray(self.values).squeeze()

    def astype(self, dtype, casting='unsafe'):
        """
        Cast values to the specified type.
        """
        return self._constructor(self.values.astype(dtype=dtype, casting=casting)).__finalize__(self)

    def compute(self):
        """
        Calculates and returns the number of records in the RDD.

        This calls the Spark count() method on the underlying RDD and updates
        the .nrecords metadata attribute.
        """
        if isinstance(self.values, BoltArray):
            self.values.tordd().count()

    def mean(self):
        """
        Mean of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def sum(self):
        """
        Sum of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def var(self):
        """
        Variance of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def std(self):
        """
        Standard deviation computed of values along the appropriate dimension.
        """
        raise NotImplementedError

    def max(self):
        """
        Maximum of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def min(self):
        """
        Minimum of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def coalesce(self, npartitions):
        """
        Coalesce data (Spark only).

        Parameters
        ----------
        npartitions : int
            Number of partitions after coalescing.
        """
        if self.mode == 'spark':
            current = self.values.tordd().getNumPartitions()
            if npartitions > current:
                raise Exception('Trying to increase number of partitions (from %g to %g), '
                                'cannot use coalesce, try repartition' % (current, npartitions))
            self.values._rdd = self.values._rdd.coalesce(npartitions)
            return self

    def cache(self):
        """
        Enable in-memory caching.
        """
        if self.mode == 'spark':
            self.values.cache()
            return self

    def repartition(self, npartitions):
        """
        Repartition data (Spark only).

        Parameters
        ----------
        npartitions : int
            Number of partitions after repartitions.
        """
        if self.mode == 'spark':
            self.values._rdd = self.values._rdd.repartition(npartitions)
            return self

    def _align(self, axes, key_shape=None):
        """
        Align local arrays so that axes for iteration are in the keys.

        This operation is applied before most functional operators.
        It ensures that the specified axes are valid, and might transpose/reshape
        the underlying array so that the functional operators can be applied
        over the correct records.

        Parameters
        ----------
        axes: tuple[int]
            One or more axes that will be iterated over by a functional operator
        """
        # ensure that the key axes are valid for an ndarray of this shape
        inshape(self.shape, axes)

        # compute the set of dimensions/axes that will be used to reshape
        remaining = [dim for dim in range(len(self.shape)) if dim not in axes]
        key_shape = key_shape if key_shape else [self.shape[axis] for axis in axes]
        remaining_shape = [self.shape[axis] for axis in remaining]
        linearized_shape = [prod(key_shape)] + remaining_shape

        # compute the transpose permutation
        transpose_order = axes + remaining

        # transpose the array so that the keys being mapped over come first, then linearize keys
        reshaped = self.values.transpose(*transpose_order).reshape(*linearized_shape)

        return reshaped

    def filter(self, func, axis=(0,)):
        """
        Filter array along an axis.

        Applies a function which should evaluate to boolean,
        along a single axis or multiple axes. Array will be
        aligned so that the desired set of axes are in the
        keys, which may require a transpose/reshape.

        Parameters
        ----------
        func : function
            Function to apply, should return boolean

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to filter along.
        """
        if self.mode == 'local':
            axes = sorted(tupleize(axis))
            reshaped = self._align(axes)
            filtered = asarray(list(filter(func, reshaped)))
            return self._constructor(filtered)

        if self.mode == 'spark':
            filtered = self.values.filter(func)
            return self._constructor(filtered, mode=self.mode)

    def map(self, func, axis=(0,), value_shape=None):
        """
        Apply a function across an axis.

        Array will be aligned so that the desired set of axes
        are in the keys, which may require a transpose/reshape.

        Parameters
        ----------
        func : function
            Function of a single array to apply

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to apply function along.
        """
        if self.mode == 'local':
            axes = sorted(tupleize(axis))
            key_shape = [self.shape[axis] for axis in axes]
            reshaped = self._align(axes, key_shape=key_shape)

            mapped = asarray(list(map(func, reshaped)))
            elem_shape = mapped[0].shape

            # invert the previous reshape operation, using the shape of the map result
            linearized_shape_inv = key_shape + list(elem_shape)
            reordered = mapped.reshape(*linearized_shape_inv)

            return self._constructor(reordered, mode=self.mode)

        if self.mode == 'spark':
            mapped = self.values.map(func, axis, value_shape)
            return self._constructor(mapped, mode=self.mode)

    def reduce(self, func, axis=0):
        """
        Reduce an array along an axis.

        Applies an associative/commutative function of two arguments
        cumulatively to all arrays along an axis. Array will be aligned
        so that the desired set of axes are in the keys, which may
        require a transpose/reshape.

        Parameters
        ----------
        func : function
            Function of two arrays that returns a single array

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to reduce along.
        """
        if self.mode == 'local':
            axes = sorted(tupleize(axis))

            # if the function is a ufunc, it can automatically handle reducing over multiple axes
            if isinstance(func, ufunc):
                inshape(self.shape, axes)
                reduced = func.reduce(self, axis=tuple(axes))
            else:
                reshaped = self._align(axes)
                reduced = reduce(func, reshaped)

            new_array = self._constructor(reduced)

            # ensure that the shape of the reduced array is valid
            expected_shape = [self.shape[i] for i in range(len(self.shape)) if i not in axes]
            if new_array.shape != tuple(expected_shape):
                raise ValueError("reduce did not yield a BoltArray with valid dimensions")

            return self._constructor(new_array, mode=self.mode)

        if self.mode == 'spark':
            reduced = self.values.reduce(func, axis)
            return self._constructor(reduced, mode=self.mode)

    def sample(self, nsamples=100, thresh=None, stat='std', seed=None):
        """
        Extract random subset of records, filtering on a summary statistic.

        Parameters
        ----------
        nsamples : int, optional, default = 100
            The number of data points to sample

        thresh : float, optional, default = None
            A threshold on statistic to use when picking points

        stat : str, optional, default = 'std'
            Statistic to use for thresholding

        Returns
        -------
        result : array
            A local numpy array with the subset of points
        """
        from numpy.linalg import norm
        from numpy import random

        statdict = {'mean': mean, 'std': std, 'max': max, 'min': min, 'norm': norm}

        if seed is None:
            seed = random.randint(0, 2 ** 32)

        if thresh is not None:
            func = statdict[stat]
            result = array(self.values.filter(
                lambda x: func(x) > thresh).takeSample(False, nsamples, seed))
        else:
            result = array(self.values.takeSample(False, nsamples, seed))

        if size(result) == 0:
            raise Exception("Nothing found, try changing '%s' threshold on '%s'" % (stat, thresh))

        return result

    def first(self):
        """
        Return the first element.
        """
        if self.mode == 'local':
            return self.values[0]

        if self.mode == 'spark':
            return self.values.tordd().values().first()
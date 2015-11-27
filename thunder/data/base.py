from numpy import asarray, maximum, minimum, add

class Data(object):
    """
    Generic base class for data types.

    All data types are backed by an collection of key-value pairs
    where the key is an integer or tuple identifier and the value is an array.

    This base class mainly provides convenience functions for accessing
    properties of the object using methods appropriate for the
    underlying computational backend.
    """
    _metadata = ['_nrecords', '_dtype', '_shape']

    def __init__(self, rdd, nrecords=None, dtype=None):
        self.rdd = rdd
        self._nrecords = nrecords
        self._dtype = dtype
        self._shape = None

    def __repr__(self):
        s = self.__class__.__name__
        for k in self._metadata:
            v = getattr(self, k)
            if v is None:
                output = 'None (inspect to compute)'
            else:
                output = str(v)
            if len(output) > 50:
                output = output[0:50].strip() + ' ... '
                if output.lstrip().startswith('['):
                    output += '] '
                if hasattr(v, '__len__'):
                    output += '(length: %d)' % len(v)
            if k.startswith('_'):
                k = k.lstrip('_')
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
            attributes found in nopropagate will *not* have their values propagated forward from the passed object,
            but will keep their existing values, even if these are None. Attribute names should be specified
            in their "private" versions (with underscores; e.g. "_dtype" and not "dtype") where applicable.
        """
        if isinstance(other, Data):
            for name in self._metadata:
                if name not in noprop:
                    otherattr = getattr(other, name, None)
                    if (otherattr is not None) and (getattr(self, name, None) is None):
                        object.__setattr__(self, name, otherattr)
        return self

    def __getitem__(self, item):
        isrange = False
        if isinstance(item, slice):
            isrange = True
        elif hasattr(item, '__iter__'):
            if any([isinstance(slise, slice) for slise in item]):
                isrange = True
        result = self.getrange(item, keys=False) if isrange else self.get(item)
        if (result is None) or (result == []):
            raise KeyError("No value found for key: %s" % str(item))
        return asarray(result)

    def _reset(self):
        self._nrecords = None
        return self

    @property
    def _constructor(self):
        return Data

    @property
    def nrecords(self):
        if self._nrecords is None:
            self._nrecords = self.rdd.count()
        return self._nrecords

    @property
    def dtype(self):
        if not self._dtype:
            self.fromfirst()
        return self._dtype

    def fromfirst(self):
        """
        Calls first() on the underlying rdd, using the returned record to determine appropriate attribute settings
        for this object (for instance, setting self.dtype to match the dtype of the underlying rdd records).

        This method is expected to be overridden by subclasses. Subclasses should first call
        super(cls, self).fromfirst(), then use the returned record to set any additional attributes.

        Returns the result of calling self.rdd.first().
        """
        from numpy import asarray
        
        record = self.rdd.first()
        self._dtype = str(asarray(record[1]).dtype)
        return record

    def first(self):
        """
        Return first record.

        This calls the Spark first() method on the underlying RDD. As a side effect, any attributes on this object that
        can be set based on the values of the first record will be set (see fromfirst).
        """
        return self.fromfirst()

    def take(self, *args, **kwargs):
        """
        Take samples.

        This calls the Spark take() method on the underlying RDD.
        """
        return self.rdd.take(*args, **kwargs)

    @staticmethod
    def _keycheck(actual, expected):
        if hasattr(actual, "__iter__"):
            try:
                speclen = len(expected) if hasattr(expected, "__len__") else \
                    reduce(lambda x, y: x + y, [1 for _ in expected], initial=0)
                if speclen != len(actual):
                    raise ValueError("Length of key specifier '%s' does not match length of first key '%s'" %
                                     (str(expected), str(actual)))
            except TypeError:
                raise ValueError("Key specifier '%s' appears not to be a sequence type, but actual keys are " %
                                 str(expected) + "sequences (first key: '%s')" % str(actual))
        else:
            if hasattr(expected, "__iter__"):
                raise ValueError("Key specifier '%s' appears to be a sequence type, " % str(expected) +
                                 "but actual keys are not (first key: '%s')" % str(actual))

    def get(self, key):
        """
        Returns single value locally to driver that matches the passed key

        If multiple records are found with keys matching the passed key, a sequence of all matching
        values will be returned. If no records found, will return None
        """
        if not hasattr(key, '__iter__'):
            key = (key,)
        record = self.first()[0]
        Data._keycheck(record, key)
        filtered = self.rdd.filter(lambda (k, v): k == key).values().collect()
        if len(filtered) == 1:
            return filtered[0]
        elif not filtered:
            return None
        else:
            return filtered

    def getmany(self, keys):
        """
        Returns values locally to driver that correspond to the passed sequence of keys.

        The return value will be a sequence equal in length to the passed keys, with each
        value in the returned sequence corresponding to the key at the same position in the passed
        keys sequence. If no value is found for a given key, the corresponding sequence element will be None.
        If multiple values are found, a sequence containing all matching values will be returned.
        """
        record = self.first()[0]
        for i, key in enumerate(keys):
            if not hasattr(key, '__iter__'):
                keys[i] = (key,)
        for key in keys:
            Data._keycheck(record, key)
        filtered = self.rdd.filter(lambda (k, _): k in frozenset(keys)).collect()
        sortvals = {}
        for k, v in filtered:
            sortvals.setdefault(k, []).append(v)
        out = []
        for k in keys:
            vals = sortvals.get(k)
            if vals is not None:
                if len(vals) == 1:
                    vals = vals[0]
            out.append(vals)
        return out

    def getrange(self, slices, keys=False):
        """
        Returns key/value pairs locally to driver that fall within a given range.

        The return values will be a sorted list of key/value pairs of all records in the underlying
        RDD for which the key falls within the range given by the passed slice selectors. Note that
        this may be very large, and could potentially exhaust the available memory on the driver.

        For singleton keys, a single slice (or slice sequence of length one) should be passed.
        For tuple keys, a sequence of multiple slices should be passed. A `step` attribute on slices
        is not supported and a alueError will be raised if passed.

        Parameters
        ----------
        slices: slice object or sequence of slices
            The passed slice or slices should be of the same cardinality as the keys of the underlying rdd.

        keys: boolean, optional, default = True
            Whether to return keys along with values, if false will just return an array of values.

        Returns
        -------
        sorted sequence of key/value pairs
        """
        if isinstance(slices, slice):
            slices = (slices,)

        def single(kv):
            key, _ = kv
            if isinstance(slices, slice):
                if slices.stop is None:
                    return key >= slices.start
                return slices.stop > key >= slices.start
            else:
                return key == slices

        def multi(kv):
            key, _ = kv
            for s, k in zip(slices, key):
                if isinstance(s, slice):
                    if s.stop is None:
                        if k < s.start:
                            return False
                    elif not (s.stop > k >= s.start):
                        return False
                else:
                    if k != s:
                        return False
            return True

        record = self.first()[0]
        Data._keycheck(record, slices)
        if not hasattr(slices, '__iter__'):
            func = single
            if hasattr(slices, 'step') and slices.step is not None:
                raise ValueError("'step' slice attribute is not supported in getRange, got step: %d" %
                                 slices.step)
        else:
            func = multi
            for s in slices:
                if hasattr(s, 'step') and s.step is not None:
                    raise ValueError("'step' slice attribute is not supported in getRange, got step: %d" %
                                     s.step)

        filtered = self.rdd.filter(func).collect()
        output = sorted(filtered)

        if keys is True:
            return output
        else:
            return map(lambda (k, v): v, output)

    def values(self):
        """
        Return rdd of values, ignoring keys

        This calls the Spark values() method on the underlying RDD.
        """
        return self.rdd.values()

    def keys(self):
        """
        Return rdd of keys, ignoring values

        This calls the Spark keys() method on the underlying RDD.
        """
        return self.rdd.keys()

    def astype(self, dtype, casting='safe'):
        """
        Cast values to specified numpy dtype.

        If 'smallfloat' is passed, values will be cast to the smallest floating point representation
        to which they can be cast safely, as determined by the thunder.utils.common smallest_float_type function.
        Typically this will be a float type larger than a passed integer type (for instance, float16 for int8 or uint8).

        If the passed dtype is the same as the current dtype, or if 'smallfloat' is passed when values are already
        in floating point, then this method will return self unchanged.

        Parameters
        ----------
        dtype: numpy dtype or dtype specifier, or string 'smallfloat', or None
            Data type to which RDD values are to be cast. Will return without cast if None is passed.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's astype() method; see numpy documentation for details.

        Returns
        -------
        New Data object, of same type as self, with values cast to the requested dtype; or self if no cast is performed.
        """
        from numpy import ndarray
        from numpy import dtype as dtypeFunc

        if dtype is None or dtype == '':
            return self

        if dtype == 'smallfloat':
            from thunder.utils.common import smallfloat
            dtype = smallfloat(self.dtype)

        def cast(v, dtype_, casting_):
            if isinstance(v, ndarray):
                return v.astype(dtype_, casting=casting_, copy=False)
            else:
                return asarray([v]).astype(dtype_, casting=casting_, copy=False)[0]

        new = self.rdd.mapValues(lambda v: cast(v, dtypeFunc(dtype), casting))
        return self._constructor(new, dtype=str(dtype)).__finalize__(self)

    def apply(self, func, keepdtype=False, keepindex=False):
        """
        Apply arbitrary function to records of a Data object.

        This wraps the combined process of calling Spark's map operation on
        the underlying RDD and returning a reconstructed Data object.

        Parameters
        ----------
        func : function
            Function to apply to records.

        keep_dtype : boolean
            Whether to preserve the dtype, if false dtype will be set to none
            under the assumption that the function might change it.

        keep_index : boolean
            Whether to preserve the index, if false index will be set to none
            under the assumption that the function might change it.
        """
        noprop = ()
        if keepdtype is False:
            noprop += ('_dtype',)
        if keepindex is False:
            noprop += ('_index',)
        return self._constructor(self.rdd.map(func)).__finalize__(self, noprop=noprop)

    def applykeys(self, func, **kwargs):
        """
        Apply arbitrary function to the keys of a Data object, preserving the values.

        See also
        --------
        Data.apply
        """

        return self.apply(lambda (k, v): (func(k), v), **kwargs)

    def applyvalues(self, func, **kwargs):
        """
        Apply arbitrary function to the values of a Data object, preserving the keys.

        See also
        --------
        Data.apply
        """
        return self.apply(lambda (k, v): (k, func(v)), **kwargs)

    def collect(self, sorting=False):
        """
        Return all records locally to the driver

        This will be slow for large datasets, and may exhaust the available memory on the driver.

        This calls the Spark collect() method on the underlying RDD.
        """
        if sorting:
            return self.sort().rdd.collect()
        else:
            return self.rdd.collect()

    def toarray(self, sorting=False):
        """
        Return all records to the driver as a numpy array

        This will be slow for large datasets, and may exhaust the available memory on the driver.
        """
        if sorting:
            rdd = self.sort().rdd
        else:
            rdd = self.rdd
        return asarray(rdd.values().collect()).squeeze()

    def sort(self):
        """
        Sort records by keys.

        This calls the Spark sortByKey() method on the underlying RDD, but reverse the order
        of the key tuples before and after sorting so they are sorted according to the convention
        that the first key varies fastest, then the second, then the third, etc.
        """
        reverse = lambda k: k[::-1] if isinstance(k, tuple) else k
        newrdd = self.rdd.map(lambda (k, v): (reverse(k), v)).sortByKey().map(
            lambda (k, v): (reverse(k), v))
        return self._constructor(newrdd).__finalize__(self)

    def count(self):
        """
        Calculates and returns the number of records in the RDD.

        This calls the Spark count() method on the underlying RDD and updates
        the .nrecords metadata attribute.
        """
        count = self.rdd.count()
        self._nrecords = count
        return count

    def mean(self, dtype='float64', casting='safe'):
        """
        Mean of values computed by aggregating across records, returned as an ndarray
        with the same size as a single record.

        If dtype is not None, then the values will first be cast to the requested
        type before the operation is performed. See Data.astype() for details.
        """
        return self.stats('mean', dtype=dtype, casting=casting).mean()

    def sum(self, dtype='float64', casting='safe'):
        """
        Sum of values computed by aggregating across records, returned as an ndarray
        with the same size as a single record.

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.sum() is equivalent to obj.astype(dtype, casting).rdd.values().sum().
        """
        out = self.astype(dtype, casting)
        return out.rdd.values().treeReduce(add, depth=3)

    def var(self, dtype='float64', casting='safe'):
        """
        Variance of values computed by aggregating across records, returned as an ndarray
        with the same size as a single record.

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.
        """
        return self.stats('var', dtype=dtype, casting=casting).var()

    def std(self, dtype='float64', casting='safe'):
        """
        Standard deviation of values computed by aggregating across records, returned as an ndarray
        with the same size as a single record.

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.stdev() is equivalent to obj.astype(dtype, casting).rdd.values().stdev().
        """
        return self.stats('std', dtype=dtype, casting=casting).std()

    def stats(self, requested='all', dtype='float64', casting='safe'):
        """
        Return a L{StatCounter} object that captures all or some of the mean, variance, maximum, minimum,
        and count of the RDD's elements in one operation.

        If dtype is specified and not None, will first cast the data as described in Data.astype().

        Parameters
        ----------
        requestedStats: sequence of one or more requested stats, or 'all'
            Possible stats include 'mean', 'sum', 'min', 'max', 'variance', 'sampleVariance', 'stdev', 'sampleStdev'.

        dtype: numpy dtype or dtype specifier, or string 'smallfloat', or None
            Data type to which RDD values are to be cast before calculating stats. See Data.astype().

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Method of casting to use. See Data.astype() and numpy astype() function.
        """
        from thunder.utils.statcounter import StatCounter

        def reducer(left_counter, right_counter):
            return left_counter.mergeStats(right_counter)

        out = self.astype(dtype, casting)
        parts = out.values().mapPartitions(lambda i: [StatCounter(i, stats=requested)])
        result = parts.treeReduce(reducer, depth=3)
        return result

    def max(self):
        """ Maximum of values across keys, returned as an ndarray. """
        return self.rdd.values().treeReduce(maximum, depth=3)

    def min(self):
        """ Minimum of values across keys, returned as an ndarray. """
        return self.rdd.values().treeReduce(minimum, depth=3)

    def coalesce(self, npartitions):
        """
        Coalesce data (used to reduce number of partitions).

        This calls the Spark coalesce() method on the underlying RDD.

        Parameters
        ----------
        numPartitions : int
            Number of partitions in coalesced RDD
        """
        current = self.rdd.getNumPartitions()
        if npartitions > current:
            raise Exception('Trying to increase number of partitions (from %g to %g), '
                            'cannot use coalesce, try repartition' % (current, npartitions))
        self.rdd = self.rdd.coalesce(npartitions)
        return self

    def cache(self):
        """
        Enable in-memory caching.

        This calls the Spark cache() method on the underlying RDD.
        """
        self.rdd.cache()
        return self

    def repartition(self, npartitions):
        """
        Repartition data.

        This calls the Spark repartition() method on the underlying RDD.

        Parameters
        ----------
        numPartitions : int
            Number of partitions in new RDD
        """
        self.rdd = self.rdd.repartition(npartitions)
        return self

    def filter(self, func):
        """
        Filter records by applying a function to each record.

        This calls the Spark filter() method on the underlying RDD.

        Parameters
        ----------
        func : function
            The function to compute on each record, should evaluate to Boolean
        """
        return self._constructor(self.rdd.filter(lambda d: func(d))).__finalize__(self)._reset()

    def filterkeys(self, func):
        """
        Filter records by applying a function to keys

        See also
        --------
        Data.filter
        """
        return self._constructor(self.rdd.filter(lambda (k, v): func(k))).__finalize__(self)._reset()

    def filtervalues(self, func):
        """
        Filter records by applying a function to values

        See also
        --------
        Data.filter
        """
        return self._constructor(self.rdd.filter(lambda (k, v): func(v))).__finalize__(self)._reset()

    def range(self, start, stop):
        """
        Extract records with keys that fall inside a range

        Returns another Data object, unlike
        getRange which returns a local array to the driver

        See also
        --------
        Data.filterOnKeys
        """
        if start >= stop:
            raise ValueError("Start (%g) is greater than or equal to stop (%g), result will be empty" % (start, stop))

        return self.filterkeys(lambda k: start <= k < stop)
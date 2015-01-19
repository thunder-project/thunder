class Data(object):
    """
    Generic base class for data types in thunder.

    All data types are backed by an RDD of key-value pairs
    where the key is a tuple identifier and the value is an array

    This base class mainly provides convenience functions for accessing
    properties of the object using the appropriate RDD methods.

    Attributes
    ----------

    rdd: Spark RDD
        The Spark Resilient Distributed Dataset wrapped by this Data object.
        Standard pyspark RDD methods on a data instance `obj` that are not already
        directly exposed by the Data object can be accessed via `obj.rdd`.
    """

    _metadata = ['_dtype']

    def __init__(self, rdd, dtype=None):
        self.rdd = rdd
        self._dtype = dtype

    def __repr__(self):

        # start with class name
        s = self.__class__.__name__

        # build a printable string by iterating through the dictionary
        for k, v in self.__dict__.iteritems():
            if k is not 'rdd':
                if v is None:
                    output = 'None (inspect to compute)'
                else:
                    output = str(v)
                # TODO make max line length a configurable property
                if len(output) > 70:
                    output = output[0:70] + ' ...'
                    if k is '_index':
                        output += '\nlength: ' + str(len(v))
                # assumes all non-rdd attributes have underscores (and drops them)
                s += '\n' + k[1:] + ': ' + output
        return s

    @property
    def dtype(self):
        if not self._dtype:
            self.populateParamsFromFirstRecord()
        return self._dtype

    def populateParamsFromFirstRecord(self):
        """Calls first() on the underlying rdd, using the returned record to determine appropriate attribute settings
        for this object (for instance, setting self.dtype to match the dtype of the underlying rdd records).

        This method is expected to be overridden by subclasses. Subclasses should first call
        super(cls, self).populateParamsFromFirstRecord(), then use the returned record to set any additional attributes.

        Returns the result of calling self.rdd.first().
        """

        from numpy import asarray
        
        record = self.rdd.first()
        self._dtype = str(asarray(record[1]).dtype)
        return record

    def __finalize__(self, other, noPropagate=()):
        """
        Lazily propagate attributes from other to self, only if attributes
        are not already defined in self

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate

        noPropagate : iterable of string attribute names, default empty tuple
            attributes found in nopropagate will *not* have their values propagated forward from the passed object,
            but will keep their existing values, even if these are None. Attribute names should be specified
            in their "private" versions (with underscores; e.g. "_dtype" and not "dtype") where applicable.
        """
        if isinstance(other, Data):
            for name in self._metadata:
                if name not in noPropagate:
                    if (getattr(other, name, None) is not None) and (getattr(self, name, None) is None):
                        object.__setattr__(self, name, getattr(other, name, None))
        return self

    @property
    def _constructor(self):
        raise NotImplementedError

    def _resetCounts(self):
        # to be overridden in subclasses
        pass

    def first(self):
        """ Return first record.

        This calls the Spark first() method on the underlying RDD. As a side effect, any attributes on this object that
        can be set based on the values of the first record will be set (see populateParamsFromFirstRecord).
        """
        return self.populateParamsFromFirstRecord()

    def take(self, *args, **kwargs):
        """ Take samples

        This calls the Spark take() method on the underlying RDD.
        """
        return self.rdd.take(*args, **kwargs)

    def values(self):
        """ Return values, ignoring keys

        This calls the Spark values() method on the underlying RDD.
        """
        return self.rdd.values()

    def keys(self):
        """ Return keys, ignoring values

        This calls the Spark keys() method on the underlying RDD.
        """
        return self.rdd.keys()

    def astype(self, dtype, casting='safe'):
        """Cast values to specified numpy dtype

        Calls numpy's astype() method.

        If the string 'smallfloat' is passed, then the values will be cast to the smallest floating point representation
        to which they can be cast safely, as determined by the thunder.utils.common smallest_float_type function.
        Typically this will be a float type larger than a passed integer type (for instance, float16 for int8 or uint8).

        If the passed dtype is the same as the current dtype, or if 'smallfloat' is passed when values are already
        in floating point, then this method will return immediately, returning self.

        Parameters
        ----------
        dtype: numpy dtype or dtype specifier, or string 'smallfloat', or None
            Data type to which RDD values are to be cast. Will return immediately, performing no cast, if None is passed.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's astype() method; see numpy documentation for details.

        Returns
        -------
        New Data object, of same type as self, with values cast to the requested dtype; or self if no cast is performed.
        """
        if dtype is None or dtype == '':
            return self
        if dtype == 'smallfloat':
            # get the smallest floating point type that can be safely cast to from our current type
            from thunder.utils.common import smallestFloatType
            dtype = smallestFloatType(self.dtype)

        nextRdd = self.rdd.mapValues(lambda v: v.astype(dtype, casting=casting, copy=False))
        return self._constructor(nextRdd, dtype=str(dtype)).__finalize__(self)

    def apply(self, func, dtype=None, casting='safe'):
        """ Apply arbitrary function to records of a Data object.

        This wraps the combined process of calling Spark's map operation on
        the underlying RDD and returning a reconstructed Data object.

        If `dtype` is passed, output will be cast to specified datatype - see `astype()`.
        Otherwise output will be assumed to be of same datatype as input.

        Parameters
        ----------
        func : function
            Function to apply to records.

        dtype: numpy dtype or dtype specifier, or string 'smallfloat', or None
            Data type to which RDD values are to be cast. Will return immediately, performing no cast, if None is passed.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's astype() method; see numpy documentation for details.
        """

        applied = self._constructor(self.rdd.map(func)).__finalize__(self)
        if dtype:
            return applied.astype(dtype=dtype, casting=casting)
        return applied

    def applyKeys(self, func):
        """ Apply arbitrary function to the keys of a Data object, preserving the values.

        See also
        --------
        Series.apply
        """

        return self._constructor(self.rdd.map(lambda (k, _): (func(k), _))).__finalize__(self)

    def applyValues(self, func, dtype=None, casting='safe'):
        """ Apply arbitrary function to the values of a Data object, preserving the keys.

        See also
        --------
        Series.apply
        """

        applied = self._constructor(self.rdd.mapValues(func)).__finalize__(self)
        if dtype:
            return applied.astype(dtype=dtype, casting=casting)
        return applied

    def collect(self):
        """ Return all records to the driver

        This will be slow for large datasets, and may exhaust the available memory on the driver.

        This calls the Spark collect() method on the underlying RDD.
        """
        return self.rdd.collect()

    def collectAsArray(self):
        """ Return all records to the driver as a numpy array

        This will be slow for large datasets, and may exhaust the available memory on the driver.
        """
        from numpy import asarray
        return asarray(self.rdd.values().collect())

    def count(self):
        """ Mean of values, ignoring keys

        This calls the Spark count() method on the underlying RDD.
        """
        return self.rdd.count()

    def mean(self, dtype='float64', casting='safe'):
        """ Mean of values, ignoring keys

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.
        """
        return self.stats('mean', dtype=dtype, casting=casting).mean()

    def sum(self, dtype='float64', casting='safe'):
        """ Sum of values, ignoring keys

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.sum() is equivalent to obj.astype(dtype, casting).rdd.values().sum().
        """
        out = self.astype(dtype, casting)
        return out.rdd.values().sum()

    def variance(self, dtype='float64', casting='safe'):
        """ Variance of values, ignoring keys

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.
        """
        return self.stats('variance', dtype=dtype, casting=casting).variance()

    def stdev(self, dtype='float64', casting='safe'):
        """ Standard deviation of values, ignoring keys

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.stdev() is equivalent to obj.astype(dtype, casting).rdd.values().stdev().
        """
        return self.stats('stdev', dtype=dtype, casting=casting).stdev()

    def stats(self, requestedStats='all', dtype='float64', casting='safe'):
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

        def redFunc(left_counter, right_counter):
            return left_counter.mergeStats(right_counter)

        out = self.astype(dtype, casting)
        return out.values().mapPartitions(lambda i: [StatCounter(i, stats=requestedStats)]).reduce(redFunc)

    def max(self):
        """ Maximum of values, ignoring keys """
        # keep using reduce(maximum) at present rather than stats('max') - stats method results in inadvertent
        # cast to float64
        from numpy import maximum
        return self.rdd.values().reduce(maximum)

    def min(self):
        """ Minimum of values, ignoring keys """
        # keep using reduce(minimum) at present rather than stats('min') - stats method results in inadvertent
        # cast to float64
        from numpy import minimum
        return self.rdd.values().reduce(minimum)

    def coalesce(self, numPartitions):
        """ Coalesce data (used to reduce number of partitions).

        This calls the Spark coalesce() method on the underlying RDD.

        Parameters
        ----------
        numPartitions : int
            Number of partitions in coalesced RDD
        """
        current = self.rdd.getNumPartitions()
        if numPartitions > current:
            raise Exception('Trying to increase number of partitions (from %g to %g), '
                            'cannot use coalesce, try repartition' % (current, numPartitions))
        self.rdd = self.rdd.coalesce(numPartitions)
        return self

    def cache(self):
        """ Enable in-memory caching.

        This calls the Spark cache() method on the underlying RDD.
        """
        self.rdd.cache()
        return self

    def repartition(self, numPartitions):
        """ Repartition data.

        This calls the Spark repartition() method on the underlying RDD.

        Parameters
        ----------
        numPartitions : int
            Number of partitions in new RDD
        """
        self.rdd = self.rdd.repartition(numPartitions)
        return self

    def filter(self, func):
        """ Filter records by appliyng a function to each record.

        This calls the Spark filter() method on the underlying RDD.
        """
        return self._constructor(self.rdd.filter(lambda d: func(d))).__finalize__(self)._resetCounts()

    def filterOnKeys(self, func):
        """ Filter records by applying a function to keys """
        return self._constructor(self.rdd.filter(lambda (k, v): func(k))).__finalize__(self)._resetCounts()

    def filterOnValues(self, func):
        """ Filter records by applying a function to values """
        return self._constructor(self.rdd.filter(lambda (k, v): func(v))).__finalize__(self)._resetCounts()

class Data(object):
    """
    Generic base class for data types in thunder.

    All data types are backed by an RDD of key-value pairs
    where the key is a tuple identifier and the value is an array

    This base class mainly provides convienience functions for accessing
    properties of the object using the appropriate RDD methods.

    Attributes
    ----------

    rdd: Spark RDD
        The Spark Resilient Distributed Dataset wrapped by this Data object.
        Standard pyspark RDD methods on a data instance `obj` that are not already
        directly exposed by the Data object can be accessed via `obj.rdd`.
    """

    _metadata = []

    def __init__(self, rdd, dtype=None):
        self.rdd = rdd
        self._dtype = str(dtype) if dtype else None

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
        record = self.rdd.first()
        self._dtype = str(record[1].dtype)
        return record

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
        if dtype == 'smallfloat':
            from thunder.utils.common import smallest_float_type
            dtype = smallest_float_type


    def collect(self):
        """ Return all records to the driver

        This will be slow for large datasets, and may exhaust the available memory on the driver.

        This calls the Spark collect() method on the underlying RDD.
        """
        return self.rdd.collect()

    def count(self):
        """ Mean of values, ignoring keys

        This calls the Spark count() method on the underlying RDD.
        """
        return self.rdd.count()

    def mean(self):
        """ Mean of values, ignoring keys

        obj.mean() is equivalent to obj.rdd.values().mean().
        """
        return self.rdd.values().mean()

    def sum(self):
        """ Sum of values, ignoring keys

        obj.sum() is equivalent to obj.rdd.values().sum().
        """
        return self.rdd.values().sum()

    def variance(self):
        """ Variance of values, ignoring keys

        obj.variance() is equivalent to obj.rdd.values().variance()."""
        return self.rdd.values().variance()

    def stdev(self):
        """ Standard deviation of values, ignoring keys

        obj.stdev() is equivalent to obj.rdd.values().stdev().
        """
        return self.rdd.values().stdev()

    def stats(self):
        """ Stats of values, ignoring keys

        obj.stats() is equivalent to obj.rdd.values().stats().
        """
        return self.rdd.values().stats()

    def max(self):
        """ Maximum of values, ignoring keys """
        from numpy import maximum
        return self.rdd.values().reduce(maximum)

    def min(self):
        """ Minimum of values, ignoring keys """
        from numpy import minimum
        return self.rdd.values().reduce(minimum)

    def cache(self):
        """ Enable in-memory caching

        This calls the Spark cache() method on the underlying RDD.
        """
        self.rdd.cache()
        return self

    def filterOnKeys(self, func):
        """ Filter records by applying a function to keys """
        return self._constructor(self.rdd.filter(lambda (k, v): func(k))).__finalize__(self)._resetCounts()

    def filterOnValues(self, func):
        """ Filter records by applying a function to values """
        return self._constructor(self.rdd.filter(lambda (k, v): func(v))).__finalize__(self)._resetCounts()

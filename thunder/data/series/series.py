from itertools import product
from numpy import array, sum, mean, median, std, size, arange, percentile,\
    asarray, zeros, corrcoef, where, unique, array_equal, delete, \
    ravel, logical_not, max, min

from ..base import Data
from ..keys import Dimensions


class Series(Data):
    """
    Distributed collection of 1d array data with axis labels.

    Backed by an RDD of key-value pairs, where the
    key is a tuple identifier, and the value is a one-dimensional array of floating-point values.
    It also has a fixed index to represent a label for each value in the arrays.
    Can optionally store and use the dimensions of the keys (min, max, and count).

    Series data will be automatically cast to a floating-point value on loading if its on-disk
    representation is integer valued.

    Parameters
    ----------

    rdd : RDD of (tuple, array) pairs
        RDD containing the series data

    index : array-like or one-dimensional list
        Values must be unique, same length as the arrays in the input data.
        Defaults to arange(len(data)) if not provided.

    dims : Dimensions
        Specify the dimensions of the keys (min, max, and count), can
        avoid computation if known in advance

    See also
    --------
    TimeSeries : a Series where the indices represent time
    """
    _metadata = Data._metadata + ['_dims', '_index']

    def __init__(self, rdd, nrecords=None, dtype=None, index=None, dims=None):
        super(Series, self).__init__(rdd, nrecords=nrecords, dtype=dtype)
        self._index = None
        if index is not None:
            self._index = index
        if dims and not isinstance(dims, Dimensions):
            try:
                dims = Dimensions.fromTuple(dims)
            except:
                raise TypeError("Series dims parameter must be castable to Dimensions object, got: %s" % str(dims))
        self._dims = dims

    @property
    def index(self):
        if self._index is None:
            self.fromfirst()
        return self._index
        
    @index.setter
    def index(self, value):
        lenself = len(self.index)
        if type(value) is str:
            value = [value]
        try:
            value[0]
        except:
            value = [value]
        try:
            lenvalue = len(value)
        except:
            raise TypeError("Index must be an object with a length")
        if lenvalue != lenself:
            raise ValueError("Length of new index '%g' must match length of original index '%g'"
                             .format(lenvalue, lenself))
        self._index = value

    @property
    def dims(self):
        from thunder.data.keys import Dimensions
        if self._dims is None:
            entry = self.fromfirst()[0]
            n = size(entry)
            d = self.rdd.keys().mapPartitions(lambda i: [Dimensions(i, n)]).reduce(lambda x, y: x.mergeDims(y))
            self._dims = d
        return self._dims

    @property
    def shape(self):
        if self._shape is None:
            self._shape = tuple(self.dims) + (size(self.index),)
        return self._shape

    @property
    def dtype(self):
        return super(Series, self).dtype

    def fromfirst(self):
        """
        Calls first() on the underlying rdd, using the returned record to determine appropriate attribute settings
        for this object (for instance, setting self.dtype to match the dtype of the underlying rdd records).

        Returns the result of calling self.rdd.first().
        """
        record = super(Series, self).fromfirst()
        if self._index is None:
            val = record[1]
            try:
                l = len(val)
            except TypeError:
                # TypeError thrown after calling len() on object with no __len__ method
                l = 1
            self._index = arange(0, l)
        return record

    @property
    def _constructor(self):
        return Series

    def _reset(self):
        self._nrecords = None
        self._dims = None
        return self

    def between(self, left, right, inclusive=True):
        """
        Select subset of values within the given index range

        Parameters
        ----------
        left : int
            Left-most index in the desired range

        right: int
            Right-most index in the desired range

        inclusive : boolean, optional, default = True
            Whether selection should include bounds
        """
        if inclusive:
            crit = lambda x: left <= x <= right
        else:
            crit = lambda x: left < x < right
        return self.select(crit)

    def select(self, crit):
        """
        Select subset of values that match a given index criterion

        Parameters
        ----------
        crit : function, list, str, int
            Criterion function to apply to indices, specific index value,
            or list of indices
        """
        import types

        # handle lists, strings, and ints
        if not isinstance(crit, types.FunctionType):
            # set("foo") -> {"f", "o"}; wrap in list to prevent:
            if isinstance(crit, basestring):
                critlist = set([crit])
            else:
                try:
                    critlist = set(crit)
                except TypeError:
                    # typically means crit is not an iterable type; for instance, crit is an int
                    critlist = set([crit])
            crit = lambda x: x in critlist

        # if only one index, return it directly or throw an error
        index = self.index
        if size(index) == 1:
            if crit(index):
                return self
            else:
                raise Exception("No indices found matching criterion")

        # determine new index and check the result
        newindex = [i for i in index if crit(i)]
        if len(newindex) == 0:
            raise Exception("No indices found matching criterion")
        if array(newindex == index).all():
            return self

        # use fast logical indexing to get the new values
        subInds = where(map(lambda x: crit(x), index))
        rdd = self.rdd.mapValues(lambda x: x[subInds])

        # if singleton, need to check whether it's an array or a scalar/int
        # if array, recompute a new set of indices
        if len(newindex) == 1:
            rdd = rdd.mapValues(lambda x: x[0])
            val = rdd.first()[1]
            if size(val) == 1:
                newindex = newindex[0]
            else:
                newindex = arange(0, size(val))

        return self._constructor(rdd, index=newindex).__finalize__(self)

    def center(self, axis=0):
        """
        Center series data by subtracting the mean
        either within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to center along, rows (0) or columns (1)
        """
        if axis == 0:
            return self.apply_values(lambda x: x - mean(x), keepindex=True)
        elif axis == 1:
            meanval = self.mean()
            return self.apply_values(lambda x: x - meanval, keepindex=True)
        else:
            raise Exception('Axis must be 0 or 1')

    def standardize(self, axis=0):
        """
        Standardize series data by dividing by the standard deviation
        either within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to standardize along, rows (0) or columns (1)
        """
        if axis == 0:
            return self.apply_values(lambda x: x / std(x), keepindex=True)
        elif axis == 1:
            stdval = self.std()
            return self.apply_values(lambda x: x / stdval, keepindex=True)
        else:
            raise Exception('Axis must be 0 or 1')

    def zscore(self, axis=0):
        """
        Zscore series data by subtracting the mean and
        dividing by the standard deviation either
        within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to zscore along, rows (0) or columns (1)
        """
        if axis == 0:
            return self.apply_values(lambda x: (x - mean(x)) / std(x), keepindex=True)
        elif axis == 1:
            stats = self.stats()
            meanval = stats.mean()
            stdval = stats.std()
            return self.apply_values(lambda x: (x - meanval) / stdval, keepindex=True)
        else:
            raise Exception('Axis must be 0 or 1')

    def squelch(self, threshold):
        """
        Set all records that do not exceed the given threhsold to 0

        Parameters
        ----------
        threshold : scalar
            Level below which to set records to zero
        """
        func = lambda x: zeros(x.shape) if max(x) < threshold else x
        return self.apply_values(func, keepdtype=True, keepindex=True)

    def correlate(self, signal, var='s'):
        """
        Correlate series data against one or many one-dimensional arrays

        Parameters
        ----------
        signal : array, or str
            Signal(s) to correlate against, can be a numpy array or a
            MAT file containing the signal as a variable

        var : str
            Variable name if loading from a MAT file
        """
        from scipy.io import loadmat

        if type(signal) is str:
            s = loadmat(signal)[var]
        else:
            s = asarray(signal)

        if s.ndim == 1:
            if size(s) != size(self.index):
                raise ValueError('Size of signal `%g` does not match size of data' % size(s))
            rdd = self.rdd.mapValues(lambda x: corrcoef(x, s)[0, 1])
            newindex = 0

        elif s.ndim == 2:
            if s.shape[1] != size(self.index):
                raise ValueError('Length of signal `%g` does not match size of data' % s.shape[1])
            rdd = self.rdd.mapValues(lambda x: array([corrcoef(x, y)[0, 1] for y in s]))
            newindex = range(0, s.shape[0])
        else:
            raise Exception('Signal to correlate with must have 1 or 2 dimensions')

        return self._constructor(rdd, dtype='float64', index=newindex).__finalize__(self)

    def series_max(self):
        """
        Compute the value maximum of each record in a Series
        """
        return self.series_stat('max')

    def series_min(self):
        """
        Compute the value minimum of each record in a Series
        """
        return self.series_stat('min')

    def series_sum(self):
        """
        Compute the value sum of each record in a Series
        """
        return self.series_stat('sum')

    def series_mean(self):
        """
        Compute the value mean of each record in a Series
        """
        return self.series_stat('mean')

    def series_median(self):
        """
        Compute the value median of each record in a Series
        """
        return self.series_stat('median')

    def series_percentile(self, q):
        """
        Compute the value percentile of each record in a Series.
        
        Parameters
        ----------
        q : scalar
            Floating point number between 0 and 100 inclusive, specifying percentile.
        """
        rdd = self.rdd.mapValues(lambda x: percentile(x, q))
        return self._constructor(rdd, index=q).__finalize__(self, noprop=('_dtype',))

    def series_std(self):
        """ Compute the value std of each record in a Series """
        return self.series_stat('stdev')

    def series_stat(self, stat):
        """
        Compute a simple statistic for each record in a Series

        Parameters
        ----------
        stat : str
            Which statistic to compute
        """
        STATS = {
            'sum': sum,
            'mean': mean,
            'median': median,
            'stdev': std,
            'max': max,
            'min': min,
            'count': size
        }
        func = STATS[stat.lower()]
        rdd = self.rdd.mapValues(lambda x: func(x))
        return self._constructor(rdd, index=stat).__finalize__(self, noprop=('_dtype',))

    def series_stats(self):
        """
        Compute many statistics for each record in a Series
        """
        rdd = self.rdd.mapValues(lambda x: array([x.size, mean(x), std(x), max(x), min(x)]))
        return self._constructor(rdd, index=['count', 'mean', 'std', 'max', 'min'])\
            .__finalize__(self, noprop=('_dtype',))

    def _check_panel(self, length):
        """
        Check that given fixed panel length evenly divides index

        Parameters
        ----------
        length : int
            Fixed length with which to subdivide index
        """
        n = len(self.index)
        if divmod(n, length)[1] != 0:
            raise ValueError("Panel length '%g' must evenly divide length of series '%g'"
                             % (length, n))
        if n == length:
            raise ValueError("Panel length '%g' cannot be length of series '%g'"
                             % (length, n))

    def mean_by_panel(self, length):
        """
        Compute the mean across fixed sized panels of each record

        Parameters
        ----------
        length : int
            Fixed length with which to subdivide
        """
        self._check_panel(length)
        func = lambda v: v.reshape(-1, length).mean(axis=0)
        rdd = self.rdd.mapValues(func)
        index = arange(0, length)
        return self._constructor(rdd, index=index).__finalize__(self)

    def group_by_panel(self, length):
        """
        Regroup each record by subdividing into fixed length panels

        Will yield a new Series with N times as many records
        as the initial Series, where N is the number of blocks
        of fixed length.

        Parameters
        ----------
        length : int
            Fixed panel length with which to subdivide
        """
        self._check_panel(length)
        n = len(self.index) / length
        tupleize = lambda k: k if isinstance(k, tuple) else (k,)
        func = lambda (k, v): zip([tupleize(k) + (i,) for i in range(0, n)], list(v.reshape(-1, length)))
        rdd = self.rdd.flatMap(func)
        index = arange(0, length)
        count = self.nrecords * n
        return self._constructor(rdd, index=index, nrecords=count).__finalize__(self)

    def subset(self, nsamples=100, thresh=None, stat='std', seed=None):
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
            result = array(self.rdd.values().filter(
                lambda x: func(x) > thresh).takeSample(False, nsamples, seed))
        else:
            result = array(self.rdd.values().takeSample(False, nsamples, seed))

        if size(result) == 0:
            raise Exception("Nothing found, try changing '%s' threshold on '%s'" % (stat, thresh))

        return result

    def _makemasks(self, index=None, level=0):
        """
        Internal function for generating masks for selecting values based on multi-index values.
    
        As all other multi-index functions will call this function, basic type-checking is also
        performed at this stage.
        """
        if index is None:
            index = self.index
        
        try:
            dims = len(array(index).shape)
            if dims == 1:
                index = array(index, ndmin=2).T
        except:
            raise TypeError('For multi-index functionality: index must be convertible to a numpy ndarray')

        try:
            index = index[:, level]
        except:
            raise ValueError("Levels must be indices into individual elements of the index")

        lenIdx = index.shape[0]
        nlevels = index.shape[1]

        combs = product(*[unique(index.T[i, :]) for i in xrange(nlevels)])
        combs = array([l for l in combs])

        masks = array([[array_equal(index[i], c) for i in xrange(lenIdx)] for c in combs])

        return zip(*[(masks[x], combs[x]) for x in xrange(len(masks)) if masks[x].any()])

    def _apply_by_index(self, function, level=0):
        """
        An internal function for applying a function to groups of values based on a multi-index

        Elements of each record are grouped according to unique value combinations of the multi-
        index across the given levels of the multi-index. Then the given function is applied
        to to each of these groups separately. If this function is many-to-one, the result
        can be recast as a Series indexed by the unique index values used for grouping.
        """

        if type(level) is int:
            level = [level]

        masks, ind = self._makemasks(index=self.index, level=level)
        bcMasks = self.rdd.ctx.broadcast(masks)
        nMasks = len(masks)
        newrdd = self.rdd.mapValues(lambda v: [array(function(v[bcMasks.value[x]])) for x in xrange(nMasks)])
        index = array(ind)
        if len(index[0]) == 1:
            index = ravel(index)
        return self._constructor(newrdd, index=index).__finalize__(self, noprop=('_dtype',))

    def select_by_index(self, val, level=0, squeeze=False, filter=False, return_mask=False):
        """
        Select or filter elements of the Series by index values (across levels, if multi-index).

        The index is a property of a Series object that assigns a value to each position within
        the arrays stored in the records of the Series. This function returns a new Series where,
        within each record, only the elements indexed by a given value(s) are retained. An index
        where each value is a list of a fixed length is referred to as a 'multi-index',
        as it provides multiple labels for each index location. Each of the dimensions in these
        sublists is a 'level' of the multi-index. If the index of the Series is a multi-index, then
        the selection can proceed by first selecting one or more levels, and then selecting one
        or more values at each level.

        Parameters:
        -----------
        val: list of lists
            Specifies the selected index values. List must contain one list for each level of the
            multi-index used in the selection. For any singleton lists, the list may be replaced
            with just the integer.

        level: list of ints, optional, default=0
            Specifies which levels in the multi-index to use when performing selection. If a single
            level is selected, the list can be replaced with an integer. Must be the same length
            as val.
        
        squeeze: bool, optional, default=False
            If True, the multi-index of the resulting Series will drop any levels that contain
            only a single value because of the selection. Useful if indices are used as unique
            identifiers.

        filter: bool, optional, default=False
            If True, selection process is reversed and all index values EXCEPT those specified
            are selected.

        return_mask: bool, optional, default=False
            If True, return the mask used to implement the selection.
        """
        try:
            level[0]
        except:
            level = [level]
        try:
            val[0]
        except:
            val = [val]
        
        remove = []
        if len(level) == 1:
            try:
                val[0][0]
            except:
                val = [val]
            if squeeze and not filter and len(val) == 1:
                remove.append(level[0])
        else:
            for i in xrange(len(val)):
                try:
                    val[i][0]
                except:
                    val[i] = [val[i]]
                if squeeze and not filter and len(val[i]) == 1:
                    remove.append(level[i])
                                
        if len(level) != len(val):
            raise ValueError("List of levels must be of same length as list of corresponding values")

        p = product(*val)
        selected = set([x for x in p])

        masks, ind = self._makemasks(index=self.index, level=level)
        nmasks = len(masks)
        masks = array([masks[x] for x in xrange(nmasks) if tuple(ind[x]) in selected])

        final_mask = masks.any(axis=0)
        if filter:
            final_mask = logical_not(final_mask)
        bcMask = self.rdd.ctx.broadcast(final_mask)
        
        newrdd = self.rdd.mapValues(lambda v: v[bcMask.value])
        indFinal = array(self.index)
        if len(indFinal.shape) == 1:
            indFinal = array(indFinal, ndmin=2).T
        indFinal = indFinal[final_mask]

        if squeeze:
            indFinal = delete(indFinal, remove, axis=1)

        if len(indFinal[0]) == 1:
            indFinal = ravel(indFinal)

        elif len(indFinal[1]) == 0:
            indFinal = arange(sum(final_mask))

        result = self._constructor(newrdd, index=indFinal).__finalize__(self)

        if return_mask:
            return result, final_mask
        else:
            return result

    def aggregate_by_index(self, function, level=0):
        """
        Aggregrate the data in each record, grouping by index values (across levels, if multi-index)
        
        For each unique value of the index, applies a function to the group of elements of the RDD indexed by that
        value. Returns an RDD indexed by those unique values. For the result to be a valid Series object, the 
        aggregating function should return a simple numeric type. Also allows selection of levels within a 
        multi-index. See selectByIndex doc for more info on indices and multi-indices.
        
        Parameters:
        -----------
        function: function
            Aggregating function to apply to Series values. Should take a list or ndarray as input and return
            a simple numeric value.
            
        level: list of ints, optional, default=0
            Specifies the levels of the multi-index to use when determining unique index values. If only a single
            level is desired, can be an int.
        """
        return self._apply_by_index(function, level=level).apply_values(
            lambda v: array(v), keepindex=True)

    def stat_by_index(self, stat, level=0):
        """
        Compute the desired statistic for each uniue index values (across levels, if multi-index)

        Parameters:
        -----------
        stat: string 
            Statistic to be computed: sum, mean, median, stdev, max, min, count
            
        level: list of ints, optional, default=0
            Specifies the levels of the multi-index to use when determining unique index values. If only a single
            level is desired, can be an int.
        """
        STATS = {
            'sum': sum,
            'mean': mean,
            'median': median,
            'stdev': std,
            'max': max,
            'min': min,
            'count': size
        }
        func = STATS[stat.lower()]
        return self.aggregate_by_index(level=level, function=func)

    def sum_by_index(self, level=0):
        """
        Compute sums of series elements for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='sum')
    
    def mean_by_index(self, level=0):
        """
        Compute means of series elements for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='mean')

    def median_by_index(self, level=0):
        """
        Compute medians of series elements for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='median')

    def std_by_index(self, level=0):
        """
        Compute means of series elements for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='stdev')

    def max_by_index(self, level=0):
        """
        Compute maximum values of series elements for each unique index value (across levels, if multi-index) 
        """
        return self.stat_by_index(level=level, stat='max')

    def min_by_index(self, level=0):
        """
        Compute minimum values of series elements for each unique index value (across level, if multi-index)
        """
        return self.stat_by_index(level=level, stat='min')

    def count_by_index(self, level=0):
        """
        Count the number of series elements for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='count')

    def pack(self, sorting=False, transpose=False, dtype=None, casting='safe'):
        """
        Pack a Series into a local array (e.g. for saving)

        This operation constructs a multidimensional numpy array from the values in this Series object,
        with indexing into the returned array as implied by the Series RDD keys. The returned numpy
        array will be local to the Spark driver; the data set should be filtered down to a reasonable
        size (such as by seriesMean(), select(), or the `selection` parameter) before attempting to
        pack() a large data set.

        Parameters
        ----------
        sorting : boolean, optional, default False
            Whether to sort the local array based on the keys. In most cases the returned array will
            already be ordered correctly, and so an explicit sorting=True is typically not necessary.

        transpose : boolean, optional, default False
            Transpose the spatial dimensions of the returned array.

        dtype: numpy dtype, dtype specifier, or string 'smallfloat'. optional, default None.
            If present, will cast the values to the requested dtype before collecting on the driver. See Data.astype()
            and numpy's astype() function for details.

        casting: casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's astype() method if dtype is given; see numpy documentation for details.

        Returns
        -------
        result: numpy array
            An array with dimensionality inferred from the RDD keys. Data in an individual Series
            value will be placed into this returned array by interpreting the Series keys as indicies
            into the returned array. The shape of the returned array will be (num time points x spatial shape).
            For instance, a series derived from 4 2d images, each 64 x 128, will have dims.count==(64, 128)
            and will pack into an array with shape (4, 64, 128). If transpose is true, the spatial dimensions
            will be reversed, so that in this example the shape of the returned array will be (4, 128, 64).
        """
        out = self

        if not (dtype is None):
            out = out.astype(dtype, casting)

        if sorting is True:
            result = out.sortByKey().values().collect()
        else:
            result = out.rdd.values().collect()

        nout = size(result[0])

        # reshape into a dense array of shape (b, x, y, z)  or (b, x, y) or (b, x)
        # where b is the number of outputs per record
        out = asarray(result).reshape(((nout,) + self.dims.count)[::-1]).T

        if transpose:
            # swap arrays so that in-memory representation matches that
            # of original input. default is to return array whose shape matches
            # that of the series dims object.
            if size(self.dims.count) == 3:
                out = out.transpose([0, 3, 2, 1])
            if size(self.dims.count) == 2:  # (b, x, y) -> (b, y, x)
                out = out.transpose([0, 2, 1])

        return out.squeeze()

    def tomatrix(self):
        """
        Convert Series to Matrix, a subclass with additional methods for matrix computations
        """
        from thunder.data.series.matrix import Matrix
        return Matrix(self.rdd).__finalize__(self)

    def totimeseries(self):
        """
        Convert Series to TimeSeries
        """
        from thunder.data.series.timeseries import TimeSeries
        return TimeSeries(self.rdd).__finalize__(self)

    def tobinary(self, path, overwrite=False):
        """
        Write data to binary files
        """
        from thunder.data.series.writers import tobinary
        tobinary(self, path, overwrite=overwrite)
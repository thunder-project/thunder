from itertools import product
from numpy import array, sum, mean, median, std, size, arange, percentile,\
    asarray, zeros, corrcoef, where, unique, array_equal, delete, \
    ravel, logical_not, max, min, unravel_index, prod, random
from bolt.utils import tupleize

from ..base import Data


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
    _metadata = Data._metadata + ['index']

    def __init__(self, values, index=None, mode='local'):
        super(Series, self).__init__(values, mode=mode)
        self._index = None
        if index is not None:
            self._index = index

    @property
    def index(self):
        if self._index is None:
            self._index = arange(self.shape[-1])
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
    def length(self):
        return len(self.index)

    @property
    def baseaxes(self):
        return tuple(range(0, len(self.shape) - 1))

    @property
    def _constructor(self):
        return Series

    def tolocal(self):
        """
        Convert to local representation.
        """
        from thunder.data.series.readers import fromarray

        if self.mode == 'local':
            raise ValueError('series already in local mode')

        return fromarray(self.toarray(), index=self.index)

    def tospark(self, engine=None):
        """
        Convert to spark representation.
        """
        from thunder.data.series.readers import fromarray

        if self.mode == 'spark':
            raise ValueError('series already in spark mode')

        return fromarray(self.toarray(), index=self.index, engine=engine)

    def sample(self, nsamples=100, seed=None):
        """
        Extract random sample of series.

        Parameters
        ----------
        nsamples : int, optional, default = 100
            The number of data points to sample.

        seed : int, optional, default = None
            Random seed.
        """
        if nsamples < 1:
            raise ValueError("number of samples must be larger than 0, got '%g'" % nsamples)

        if seed is None:
            seed = random.randint(0, 2 ** 32)

        if self.mode == 'spark':
            result = asarray(self.values.tordd().values().takeSample(False, nsamples, seed))

        else:
            basedims = [self.shape[d] for d in self.baseaxes]
            inds = [unravel_index(int(k), basedims) for k in random.rand(nsamples) * prod(basedims)]
            result = asarray([self.values[tupleize(i) + (slice(None, None),)] for i in inds])

        return self._constructor(result, index=self.index)

    def map(self, func, index=None):
        """
        Map a function on each series
        """
        if index is None:
            index = self.index
        new = self._map(func, axis=self.baseaxes, value_shape=len(index))
        return self._constructor(new.values, index=index)

    def filter(self, func):
        """
        Filter by applying a function to each series.
        """
        return self._filter(func, axis=self.baseaxes)
    
    def reduce(self, func):
        """
        Reduce over series.
        """
        return self._reduce(func, axis=self.baseaxes)

    def mean(self):
        """
        Compute the mean across images
        """
        return self._constructor(self.values.mean(axis=self.baseaxes, keepdims=True))

    def var(self):
        """
        Compute the variance across images
        """
        return self._constructor(self.values.var(axis=self.baseaxes, keepdims=True))

    def std(self):
        """
        Compute the standard deviation across images
        """
        return self._constructor(self.values.std(axis=self.baseaxes, keepdims=True))

    def sum(self):
        """
        Compute the sum across images
        """
        return self._constructor(self.values.sum(axis=self.baseaxes, keepdims=True))

    def max(self):
        """
        Compute the max across images
        """
        return self._constructor(self.values.max(axis=self.baseaxes, keepdims=True))

    def min(self):
        """
        Compute the min across images
        """
        return self._constructor(self.values.min(axis=self.baseaxes, keepdims=True))

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
            Criterion function to map to indices, specific index value,
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
            if crit(index[0]):
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
        subinds = where(map(lambda x: crit(x), index))
        new = self.map(lambda x: x[subinds], index=newindex)

        # if singleton, need to check whether it's an array or a scalar/int
        # if array, recompute a new set of indices
        if len(newindex) == 1:
            new = new.map(lambda x: x[0], index=newindex)
            val = new.first()
            if size(val) == 1:
                newindex = [newindex[0]]
            else:
                newindex = arange(0, size(val))

        new._index = newindex

        return new

    def center(self, axis=1):
        """
        Center series data by subtracting the mean
        either within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to center along, within (1) or across (0) records
        """
        if axis == 1:
            return self.map(lambda x: x - mean(x))
        elif axis == 0:
            meanval = self.mean().toarray()
            return self.map(lambda x: x - meanval)
        else:
            raise Exception('Axis must be 0 or 1')

    def standardize(self, axis=1):
        """
        Standardize series data by dividing by the standard deviation
        either within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to standardize along, within (1) or across (0) records
        """
        if axis == 1:
            return self.map(lambda x: x / std(x))
        elif axis == 0:
            stdval = self.std().toarray()
            return self.map(lambda x: x / stdval)
        else:
            raise Exception('Axis must be 0 or 1')

    def zscore(self, axis=1):
        """
        Zscore series data by subtracting the mean and
        dividing by the standard deviation either
        within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to zscore along, within (1) or across (0) records
        """
        if axis == 1:
            return self.map(lambda x: (x - mean(x)) / std(x))
        elif axis == 0:
            meanval = self.mean().toarray()
            stdval = self.std().toarray()
            return self.map(lambda x: (x - meanval) / stdval)
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
        return self.map(func)

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

            return self.map(lambda x: corrcoef(x, s)[0, 1], index=[1])

        elif s.ndim == 2:
            if s.shape[1] != size(self.index):
                raise ValueError('Length of signal `%g` does not match size of data' % s.shape[1])
            newindex = arange(0, s.shape[0])
            return self.map(lambda x: array([corrcoef(x, y)[0, 1] for y in s]), index=newindex)

        else:
            raise Exception('Signal to correlate with must have 1 or 2 dimensions')

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
        return self.map(lambda x: percentile(x, q), index=[q])

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
        return self.map(lambda x: func(x), index=[stat])

    def series_stats(self):
        """
        Compute many statistics for each record in a Series
        """
        newindex = ['count', 'mean', 'std', 'max', 'min']
        return self.map(lambda x: array([x.size, mean(x), std(x), max(x), min(x)]),
                          index=newindex)

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
        newindex = arange(0, length)
        return self.map(func, index=newindex)

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
            raise TypeError('A multi-index must be convertible to a numpy ndarray')

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

    def _map_by_index(self, function, level=0):
        """
        An internal function for maping a function to groups of values based on a multi-index

        Elements of each record are grouped according to unique value combinations of the multi-
        index across the given levels of the multi-index. Then the given function is applied
        to to each of these groups separately. If this function is many-to-one, the result
        can be recast as a Series indexed by the unique index values used for grouping.
        """

        if type(level) is int:
            level = [level]

        masks, ind = self._makemasks(index=self.index, level=level)
        nMasks = len(masks)
        newindex = array(ind)
        if len(newindex[0]) == 1:
            newindex = ravel(newindex)
        return self.map(lambda v: [array(function(v[masks[x]])) for x in xrange(nMasks)],
                          index=newindex)

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
        val : list of lists
            Specifies the selected index values. List must contain one list for each level of the
            multi-index used in the selection. For any singleton lists, the list may be replaced
            with just the integer.

        level : list of ints, optional, default=0
            Specifies which levels in the multi-index to use when performing selection. If a single
            level is selected, the list can be replaced with an integer. Must be the same length
            as val.
        
        squeeze : bool, optional, default=False
            If True, the multi-index of the resulting Series will drop any levels that contain
            only a single value because of the selection. Useful if indices are used as unique
            identifiers.

        filter : bool, optional, default=False
            If True, selection process is reversed and all index values EXCEPT those specified
            are selected.

        return_mask : bool, optional, default=False
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
            raise ValueError("List of levels must be same length as list of corresponding values")

        p = product(*val)
        selected = set([x for x in p])

        masks, ind = self._makemasks(index=self.index, level=level)
        nmasks = len(masks)
        masks = array([masks[x] for x in xrange(nmasks) if tuple(ind[x]) in selected])

        final_mask = masks.any(axis=0)
        if filter:
            final_mask = logical_not(final_mask)

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

        result = self.map(lambda v: v[final_mask], index=indFinal)

        if return_mask:
            return result, final_mask
        else:
            return result

    def aggregate_by_index(self, function, level=0):
        """
        Aggregrate data in each record, grouping by index values.
        
        For each unique value of the index, applies a function to the group
        indexed by that value. Returns a Series indexed by those unique values.
        For the result to be a valid Series object, the aggregating function should
        return a simple numeric type. Also allows selection of levels within a
        multi-index. See select_by_index for more info on indices and multi-indices.
        
        Parameters:
        -----------
        function : function
            Aggregating function to map to Series values. Should take a list or ndarray
            as input and return a simple numeric value.
            
        level : list of ints, optional, default=0
            Specifies the levels of the multi-index to use when determining unique index values.
            If only a single level is desired, can be an int.
        """
        return self._map_by_index(function, level=level).map(lambda v: array(v))

    def stat_by_index(self, stat, level=0):
        """
        Compute the desired statistic for each uniue index values (across levels, if multi-index)

        Parameters:
        -----------
        stat : string
            Statistic to be computed: sum, mean, median, stdev, max, min, count
            
        level : list of ints, optional, default=0
            Specifies the levels of the multi-index to use when determining unique index values.
            If only a single level is desired, can be an int.
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
        Compute sums for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='sum')
    
    def mean_by_index(self, level=0):
        """
        Compute means for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='mean')

    def median_by_index(self, level=0):
        """
        Compute medians for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='median')

    def std_by_index(self, level=0):
        """
        Compute means for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='stdev')

    def max_by_index(self, level=0):
        """
        Compute maximum values for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='max')

    def min_by_index(self, level=0):
        """
        Compute minimum values for each unique index value (across level, if multi-index)
        """
        return self.stat_by_index(level=level, stat='min')

    def count_by_index(self, level=0):
        """
        Count the number for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='count')

    def tomatrix(self):
        """
        Convert Series to Matrix, a subclass with methods for matrix computations.
        """
        from thunder.data.series.matrix import Matrix
        return Matrix(self.values, index=self.index)

    def totimeseries(self):
        """
        Convert Series to TimeSeries, a subclass for time series computation.
        """
        from thunder.data.series.timeseries import TimeSeries
        return TimeSeries(self.values, index=self.index)

    def toblocks(self, size="150M"):
        """
        Convert Series to Blocks.

        Parameters
        ----------
        size : str, optional, default = "150M"
            String interpreted as memory size.
        """
        from thunder.data.blocks.strategy import SeriesBlockingStrategy
        if isinstance(size, SeriesBlockingStrategy):
            strategy = size
        elif isinstance(size, basestring) or isinstance(size, int):
            strategy = SeriesBlockingStrategy.generateFromBlockSize(self, size)
        else:
            strategy = SeriesBlockingStrategy(size)

        strategy.setsource(self)
        block_class = strategy.getclass()
        blocked = self.rdd.map(strategy.blocker)
        grouped = blocked.groupByKey(numPartitions=strategy.nblocks)
        rdd = grouped.map(strategy.combiner)
        return block_class(rdd, dims=self.dims, nimages=len(self.index), dtype=self.dtype)

    def toimages(self, size="150M"):
        """
        Converts Series to Images.

        Equivalent to calling series.toBlocks(size).toImages()

        Parameters
        ----------
        size : str, optional, default = "150M"
            String interpreted as memory size.
        """
        return self.toblocks(size).toimages()

    def tobinary(self, path, overwrite=False):
        """
        Write data to binary files.

        Parameters
        ----------
        path : string path or URI to directory to be created
            Output files will be written underneath path.
            Directory will be created as a result of this call.

        overwrite : bool
            If true, path and all its contents will be deleted and
            recreated as partof this call.
        """
        from thunder.data.series.writers import tobinary
        tobinary(self, path, overwrite=overwrite)
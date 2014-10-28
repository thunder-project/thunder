from numpy import ndarray, array, sum, mean, std, size, arange, \
    polyfit, polyval, percentile, asarray, maximum, zeros, corrcoef, where

from thunder.rdds.data import Data
from thunder.rdds.keys import Dimensions
from thunder.utils.common import checkparams, loadmatvar


class Series(Data):
    """
    Distributed collection of 1d array data with axis labels.

    Backed by an RDD of key-value pairs, where the
    key is a tuple identifier, and the value is a one-dimensional array.
    It also has a fixed index to represent a label for each value in the arrays.
    Can optionally store and use the dimensions of the keys (min, max, and count).

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
    SpatialSeries : a Series where the keys represent spatial coordinates
    """

    _metadata = ['_index', '_dims', '_dtype']

    def __init__(self, rdd, index=None, dims=None, dtype=None):
        super(Series, self).__init__(rdd, dtype=dtype)
        self._index = index
        if dims and not isinstance(dims, Dimensions):
            raise TypeError("Series dims parameter must be Dimensions object, got: %s" % type(dims))
        else:
            self._dims = dims

    @property
    def index(self):
        if self._index is None:
            self.populateParamsFromFirstRecord()
        return self._index

    @property
    def dims(self):
        from thunder.rdds.keys import Dimensions
        if self._dims is None:
            entry = self.populateParamsFromFirstRecord()[0]
            n = size(entry)
            d = self.rdd.keys().mapPartitions(lambda i: [Dimensions(i, n)]).reduce(lambda x, y: x.mergedims(y))
            self._dims = d
        return self._dims

    @property
    def dtype(self):
        # override just calls superclass; here for explicitness
        return super(Series, self).dtype

    def populateParamsFromFirstRecord(self):
        """Calls first() on the underlying rdd, using the returned record to determine appropriate attribute settings
        for this object (for instance, setting self.dtype to match the dtype of the underlying rdd records).

        Returns the result of calling self.rdd.first().
        """
        record = super(Series, self).populateParamsFromFirstRecord()
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

    @staticmethod
    def _check_type(record):
        key = record[0]
        value = record[1]
        if not isinstance(key, tuple):
            raise Exception('Keys must be tuples')
        if not isinstance(value, ndarray):
            raise Exception('Values must be ndarrays')
        else:
            if value.ndim != 1:
                raise Exception('Values must be 1d arrays')

    def _resetCounts(self):
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
        subinds = where(map(lambda x: crit(x), index))
        rdd = self.rdd.mapValues(lambda x: x[subinds])

        # convert an array with one value to a scalar/int
        if len(newindex) == 1:
            newindex = newindex[0]
            rdd = rdd.mapValues(lambda x: x[0])

        return self._constructor(rdd, index=newindex).__finalize__(self)

    def detrend(self, method='linear', **kwargs):
        """
        Detrend series data with linear or nonlinear detrending
        Preserve intercept so that subsequent steps can adjust the baseline

        Parameters
        ----------
        method : str, optional, default = 'linear'
            Detrending method

        order : int, optional, default = 5
            Order of polynomial, for non-linear detrending only
        """
        checkparams(method, ['linear', 'nonlin'])

        if method.lower() == 'linear':
            order = 1
        else:
            if 'order' in kwargs:
                order = kwargs['order']
            else:
                order = 5

        def func(y):
            x = arange(1, len(y)+1)
            p = polyfit(x, y, order)
            p[-1] = 0
            yy = polyval(p, x)
            return y - yy

        return self.apply(func)

    def normalize(self, baseline='percentile', **kwargs):
        """ Normalize each record in series data by
        subtracting and dividing by a baseline

        Parameters
        ----------
        baseline : str, optional, default = 'percentile'
            Quantity to use as the baseline

        perc : int, optional, default = 20
            Percentile value to use, for 'percentile' baseline only
        """
        checkparams(baseline, ['mean', 'percentile'])

        if baseline.lower() == 'mean':
            basefunc = mean
        if baseline.lower() == 'percentile':
            if 'percentile' in kwargs:
                perc = kwargs['percentile']
            else:
                perc = 20
            basefunc = lambda x: percentile(x, perc)

        def get(y):
            b = basefunc(y)
            return (y - b) / (b + 0.1)

        return self.apply(get)

    def center(self, axis=0):
        """ Center series data by subtracting the mean
        either within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to center along, rows (0) or columns (1)
        """
        if axis == 0:
            return self.apply(lambda x: x - mean(x))
        elif axis == 1:
            meanvec = self.mean()
            return self.apply(lambda x: x - meanvec)
        else:
            raise Exception('Axis must be 0 or 1')

    def standardize(self, axis=0):
        """ Standardize series data by dividing by the standard deviation
        either within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to standardize along, rows (0) or columns (1)
        """
        if axis == 0:
            return self.apply(lambda x: x / std(x))
        elif axis == 1:
            stdvec = self.stdev()
            return self.apply(lambda x: x / stdvec)
        else:
            raise Exception('Axis must be 0 or 1')

    def zscore(self, axis=0):
        """ Zscore series data by subtracting the mean and
        dividing by the standard deviation either
        within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to zscore along, rows (0) or columns (1)
        """
        if axis == 0:
            return self.apply(lambda x: (x - mean(x)) / std(x))
        elif axis == 1:
            stats = self.stats()
            meanvec = stats.mean()
            stdvec = stats.stdev()
            return self.apply(lambda x: (x - meanvec) / stdvec)
        else:
            raise Exception('Axis must be 0 or 1')

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

        # handle the case of a 1d signal
        if s.ndim == 1:
            if size(s) != size(self.index):
                raise Exception('Size of signal to correlate with, %g, does not match size of series' % size(s))
            rdd = self.rdd.mapValues(lambda x: corrcoef(x, s)[0, 1])
            newindex = 0
        # handle multiple 1d signals
        elif s.ndim == 2:
            if s.shape[1] != size(self.index):
                raise Exception('Length of signals to correlate with, %g, does not match size of series' % s.shape[1])
            rdd = self.rdd.mapValues(lambda x: array([corrcoef(x, y)[0, 1] for y in s]))
            newindex = range(0, s.shape[0])
        else:
            raise Exception('Signal to correlate with must have 1 or 2 dimensions')

        # return result
        return self._constructor(rdd, index=newindex).__finalize__(self)

    def apply(self, func):
        """ Apply arbitrary function to values of a Series,
        preserving keys and indices

        Parameters
        ----------
        func : function
            Function to apply
        """
        rdd = self.rdd.mapValues(func)
        return self._constructor(rdd, index=self._index).__finalize__(self)

    def seriesMax(self):
        """ Compute the value maximum of each record in a Series """
        return self.seriesStat('max')

    def seriesMin(self):
        """ Compute the value minimum of each record in a Series """
        return self.seriesStat('min')

    def seriesSum(self):
        """ Compute the value sum of each record in a Series """
        return self.seriesStat('sum')

    def seriesMean(self):
        """ Compute the value mean of each record in a Series """
        return self.seriesStat('mean')

    def seriesStdev(self):
        """ Compute the value std of each record in a Series """
        return self.seriesStat('stdev')

    def seriesStat(self, stat):
        """ Compute a simple statistic for each record in a Series

        Parameters
        ----------
        stat : str
            Which statistic to compute
        """
        STATS = {
            'sum': sum,
            'mean': mean,
            'stdev': std,
            'max': max,
            'min': min,
            'count': size
        }
        func = STATS[stat]
        rdd = self.rdd.mapValues(lambda x: func(x))
        return self._constructor(rdd, index=stat).__finalize__(self)

    def seriesStats(self):
        """
        Compute a collection of statistics for each record in a Series
        """
        rdd = self.rdd.mapValues(lambda x: array([x.size, mean(x), std(x), max(x), min(x)]))
        return self._constructor(rdd, index=['count', 'mean', 'std', 'max', 'min']).__finalize__(self)

    def maxProject(self, axis=0):
        """
        Project along one of the keys
        """
        import copy
        dims = copy.copy(self.dims)
        nkeys = len(self.first()[0])
        if axis > nkeys - 1:
            raise IndexError('only %g keys, cannot compute maximum along axis %g' % (nkeys, axis))
        rdd = self.rdd.map(lambda (k, v): (tuple(array(k)[arange(0, nkeys) != axis]), v)).reduceByKey(maximum)
        dims.min = list(array(dims.min)[arange(0, nkeys) != axis])
        dims.max = list(array(dims.max)[arange(0, nkeys) != axis])
        return self._constructor(rdd, dims=dims).__finalize__(self)

    def subtoind(self, order='F', onebased=True):
        """
        Convert subscript index keys to linear index keys

        Parameters
        ----------
        dims : array-like
            Maximum dimensions

        order : str, 'C' or 'F', default = 'F'
            Specifies row-major or column-major array indexing. See numpy.ravel_multi_index.

        onebased : boolean, default = True
            True if subscript indices start at 1, False if they start at 0
        """
        from thunder.rdds.keys import _subtoind_converter

        # converter = _subtoind_converter(self.dims.max, order=order, onebased=onebased)
        converter = _subtoind_converter(self.dims.count, order=order, onebased=onebased)
        rdd = self.rdd.map(lambda (k, v): (converter(k), v))
        return self._constructor(rdd, index=self._index).__finalize__(self)

    def indtosub(self, order='F', onebased=True, dims=None):
        """
        Convert linear indexing to subscript indexing

        Parameters
        ----------
        dims : array-like, optional
            Maximum dimensions. If not provided, will use dims property.

        order : str, 'C' or 'F', default = 'F'
            Specifies row-major or column-major array indexing. See numpy.unravel_index.

        onebased : boolean, default = True
            True if generated subscript indices are to start at 1, False to start at 0
        """
        from thunder.rdds.keys import _indtosub_converter

        if dims is None:
            dims = self.dims.max

        converter = _indtosub_converter(dims, order=order, onebased=onebased)
        rdd = self.rdd.map(lambda (k, v): (converter(k), v))
        return self._constructor(rdd, index=self._index).__finalize__(self)

    def pack(self, selection=None, sorting=False, transpose=False):
        """
        Pack a Series into a local array (e.g. for saving)

        This operation constructs a multidimensional numpy array from the values in this Series object,
        with indexing into the returned array as implied by the Series RDD keys. The returned numpy
        array will be local to the Spark driver; the data set should be filtered down to a reasonable
        size (such as by seriesMean(), select(), or the `selection` parameter) before attempting to
        pack() a large data set.

        Parameters
        ----------
        selection : function, list, str, or int, optional, default None
            Criterion for selecting a subset, list, or index value

        sorting : boolean, optional, default False
            Whether to sort the local array based on the keys. In most cases the returned array will
            already be ordered correctly, and so an explicit sorting=True is typically not necessary.

        transpose : boolean, optional, default False
            Transpose the spatial dimensions of the returned array.

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

        if selection:
            out = self.select(selection)
        else:
            out = self

        result = out.rdd.map(lambda (_, v): v).collect()
        nout = size(result[0])

        if sorting is True:
            keys = out.subtoind().rdd.map(lambda (k, _): int(k)).collect()
            result = array([v for (k, v) in sorted(zip(keys, result), key=lambda (k, v): k)])

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

    def subset(self, nsamples=100, thresh=None, stat='std'):
        """Extract random subset of records from a Series,
        filtering on the standard deviation

        Parameters
        ----------
        nsamples : int, optional, default = 100
            The number of data points to sample

        thresh : float, optional, default = None
            A threshold on standard deviation to use when picking points

        stat : str, optional, default = 'std
            Statistic to use for thresholding

        Returns
        -------
        result : array
            A local numpy array with the subset of points
        """
        from numpy.linalg import norm
        from numpy.random import randint

        stat_dict = {'std': std, 'norm': norm}
        seed = randint(0, 2 ** 32 - 1)

        if thresh is not None:
            func = stat_dict[stat]
            result = array(self.rdd.values().filter(lambda x: func(x) > thresh).takeSample(False, nsamples, seed=seed))
        else:
            result = array(self.rdd.values().takeSample(False, nsamples, seed=seed))

        if size(result) == 0:
            raise Exception('No records found, maybe threshold of %g is too high, try changing it?' % thresh)

        return result

    def query(self, inds, var='inds', order='F', onebased=True):
        """
        Extract records with indices matching those provided

        Keys will be automatically linearized before matching to provided indices.
        This will not affect

        Parameters
        ----------
        inds : str, or array-like (2D)
            Array of indices, each an array-like of integer indices, or
            filename of a MAT file containing a set of indices as a cell array

        var : str, optional, default = 'inds'
            Variable name if loading from a MAT file

        order : str, optional, default = 'F'
            Specify ordering for linearizing indices (see subtoind)

        onebased : boolean, optional, default = True
            Specify zero or one based indexing for linearizing (see subtoind)

        Returns
        -------
        keys : array, shape (n, k) where k is the length of each value
            Averaged values

        values : array, shape (n, d) where d is the number of keys
            Averaged keys
        """

        if isinstance(inds, str):
            inds = loadmatvar(inds, var)[0]
        else:
            inds = asarray(inds)

        n = len(inds)

        from thunder.rdds.keys import _indtosub_converter
        converter = _indtosub_converter(dims=self.dims.max, order=order, onebased=onebased)

        keys = zeros((n, len(self.dims.count)))
        values = zeros((n, len(self.first()[1])))

        data = self.subtoind(order=order, onebased=onebased)

        for idx, indlist in enumerate(inds):
            if len(indlist) > 0:
                inds_set = set(indlist.flat)
                inds_bc = self.rdd.context.broadcast(inds_set)
                values[idx, :] = data.filterOnKeys(lambda k: k in inds_bc.value).values().mean()
                keys[idx, :] = mean(map(lambda k: converter(k), indlist), axis=0)

        return keys, values

    def toRowMatrix(self):
        """
        Convert Series to RowMatrix
        """
        from thunder.rdds.matrices import RowMatrix
        return RowMatrix(self.rdd).__finalize__(self)

    def toTimeSeries(self):
        """
        Convert Series to TimeSeries
        """
        from thunder.rdds.timeseries import TimeSeries
        return TimeSeries(self.rdd).__finalize__(self)

    def toSpatialSeries(self):
        """
        Convert Series to SpatialSeries
        """
        from thunder.rdds.spatialseries import SpatialSeries
        return SpatialSeries(self.rdd).__finalize__(self)



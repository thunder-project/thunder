import os
import glob
import json
import types
from numpy import ndarray, frombuffer, dtype, array, sum, mean, std, size, arange, polyfit, polyval, percentile, load
from scipy.io import loadmat
from thunder.rdds.data import Data
from thunder.utils.common import checkparams


class Series(Data):
    """A series backed by an RDD of (tuple,array) pairs
    where the tuple is an identifier for each record,
    and each value is an array indexed by a
    common, fixed list (e.g. a time series)"""

    def __init__(self, rdd, index=None):
        super(Series, self).__init__(rdd)
        if index is not None:
            self.index = index
        else:
            record = self.rdd.first()
            self.index = arange(0, len(record[1]))

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
        crit : function
            Criterion function to apply to indices
        """
        index = self.index

        if not isinstance(crit, types.FunctionType):
            if isinstance(crit, list):
                critlist = set(crit)
            else:
                critlist = {crit}
            crit = lambda x: x in critlist

        newindex = [i for i in index if crit(i)]
        if len(newindex) == 0:
            raise Exception("No indices found matching criterion")
        if newindex == index:
            return self

        rdd = self.rdd.mapValues(lambda x: array([y[0] for y in zip(array(x), index) if crit(y[1])]))

        return Series(rdd, index=newindex)

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

    def center(self):
        """ Center series data by subtracting the mean """
        return self.apply(lambda x: x - mean(x))

    def normalize(self, baseline='percentile', **kwargs):
        """ Normalize series data by subtracting and dividing
        by a baseline

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

        def func(y):
            baseline = basefunc(y)
            return (y - baseline) / (baseline + 0.1)

        return self.apply(func)

    def standardize(self):
        """ Standardize series data by dividing by the standard deviation """
        return self.apply(lambda x: x / std(x))

    def zscore(self):
        """ Zscore series data by subtracting the mean and
        dividing by the standard deviation """
        return self.apply(lambda x: (x - mean(x)) / std(x))

    def apply(self, func):
        """ Apply arbitrary function to values of a Series,
        preserving keys and indices

        Parameters
        ----------
        func : function
            Function to apply
        """
        rdd = self.rdd.mapValues(lambda x: func(x))
        return Series(rdd, index=self.index)

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
        return Series(rdd, index=[stat])

    def seriesStats(self):
        """ Compute a collection of statistics for each record in a Series """
        rdd = self.rdd.mapValues(lambda x: array([x.size, mean(x), std(x), max(x), min(x)]))
        return Series(rdd, index=['size', 'mean', 'std', 'max', 'min'])


class SeriesLoader(object):

    def __init__(self, nkeys, nvalues, keytype='int16', valuetype='int16', minPartitions=None):
        self.nkeys = nkeys
        self.nvalues = nvalues
        self.keytype = keytype
        self.valuetype = valuetype
        self.minPartitions = minPartitions

    def fromText(self, datafile, sc):

        if os.path.isdir(datafile):
            files = sorted(glob.glob(os.path.join(datafile, '*.txt')))
            datafile = ''.join([files[x] + ',' for x in range(0, len(files))])[0:-1]

        def parse(line, nkeys):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys:])
            keys = tuple(int(x) for x in vec[:nkeys])
            return keys, ts

        lines = sc.textFile(datafile, self.minPartitions)
        nkeys = self.nkeys
        data = lines.map(lambda x: parse(x, nkeys))

        return Series(data)

    def fromBinary(self, datafile, sc):

        if os.path.isdir(datafile):
            datafile = os.path.join(datafile, '*.bin')

        keysize = self.nkeys * dtype(self.keytype).itemsize
        recordsize = keysize + self.nvalues * dtype(self.valuetype).itemsize

        lines = sc.newAPIHadoopFile(datafile, 'thunder.util.io.hadoop.FixedLengthBinaryInputFormat',
                                              'org.apache.hadoop.io.LongWritable',
                                              'org.apache.hadoop.io.BytesWritable',
                                              conf={'recordLength': str(recordsize)})

        def _parseKeysFromBinaryBuffer(buf, keydtype, keybufsize):
            return frombuffer(buffer(buf, 0, keybufsize), dtype=keydtype)

        def _parseValsFromBinaryBuffer(buf, valsdtype, keybufsize):
            return frombuffer(buffer(buf, keybufsize), dtype=valsdtype)

        keydtype = dtype(self.keytype)
        valdtype = dtype(self.valuetype)
        data = lines.map(lambda (_, v):
                         (tuple(_parseKeysFromBinaryBuffer(v, keydtype, keysize)),
                          _parseValsFromBinaryBuffer(v, valdtype, keysize)))

        return Series(data)

    def fromMatLocal(self, datafile, varname, keyfile=None):

        data = loadmat(datafile)[varname]
        if data.ndim > 2:
            raise IOError('Input data must be one or two dimensional')
        if keyfile:
            keys = map(lambda x: tuple(x), loadmat(keyfile)['keys'])
        else:
            keys = arange(0, data.shape[0])

        rdd = Series(self.sc.parallelize(zip(keys, data), self.minPartitions))

        return rdd

    def fromNpyLocal(self, datafile, keyfile=None):

        data = load(datafile)
        if data.ndim > 2:
            raise IOError('Input data must be one or two dimensional')
        if keyfile:
            keys = map(lambda x: tuple(x), load(keyfile))
        else:
            keys = arange(0, data.shape[0])

        rdd = Series(self.sc.parallelize(zip(keys, data), self.minPartitions))

        return rdd

    @staticmethod
    def loadConf(datafile, conffile='conf.json'):

        if not os.path.isfile(conffile):
            if os.path.isdir(datafile):
                basepath = datafile
            else:
                basepath = os.path.dirname(datafile)
            conffile = os.path.join(basepath, conffile)

        try:
            f = open(conffile, 'r')
            params = json.load(f)
        except IOError:
            params = None
        return params


from collections import namedtuple
import json
import types
from numpy import ndarray, frombuffer, dtype, array, sum, mean, std, size, arange, polyfit, polyval, percentile, load
from scipy.io import loadmat
import urlparse
from thunder.rdds.readers import getFileReaderForPath
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
            critlist = set(crit)
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
        return Series(rdd, index=['count', 'mean', 'std', 'max', 'min'])


class SeriesLoader(object):

    def __init__(self, sparkcontext, minPartitions=None):
        self.sc = sparkcontext
        self.minPartitions = minPartitions

    @staticmethod
    def __normalizeDatafilePattern(datapath, ext):
        parseresult = urlparse.urlparse(datapath)
        if parseresult.scheme:
            # this appears to already be a fully-qualified URI
            return datapath
        if datapath.endswith(ext) or "*" in datapath:
            # this appears to already be a filename given with the specified extension or a wildcard expression
            return "file://" + datapath
        # this appears to be a directory or similar. add wildcard match to specified extension
        return "file://" + datapath + "/*." + ext

    def fromText(self, datafile, nkeys=None, ext="txt"):
        """
        Loads Series data from text files.

        Parameters
        ----------
        datafile : string
            Specifies the file or files to be loaded. Datafile may be either a URI (with scheme specified) or a path
            on the local filesystem.
            If a path is passed (determined by the absence of a scheme component when attempting to parse as a URI),
            and it is not already a wildcard expression and does not end in <ext>, then it will be converted into a
            wildcard pattern by appending '/*.ext'. This conversion can be avoided by passing a "file://" URI.
        """
        datafile = self.__normalizeDatafilePattern(datafile, ext)

        def parse(line, nkeys_):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys_:])
            keys = tuple(int(x) for x in vec[:nkeys_])
            return keys, ts

        lines = self.sc.textFile(datafile, self.minPartitions)
        data = lines.map(lambda x: parse(x, nkeys))

        return Series(data)

    BinaryLoadParameters = namedtuple('BinaryLoadParameters', 'nkeys nvalues keyformat format')
    BinaryLoadParameters.__new__.__defaults__ = (None, None, 'int16', 'int16')

    @staticmethod
    def __loadParametersAndDefaults(datafile, conffilename, nkeys, nvalues, keytype, valuetype):
        """Collects parameters to use for binary series loading.

        Priority order is as follows:
        1. parameters specified as keyword arguments;
        2. parameters specified in a conf.json file on the local filesystem;
        3. default parameters

        Returns
        -------
        BinaryLoadParameters instance
        """
        params = SeriesLoader.loadConf(datafile, conffile=conffilename)

        # filter dict to include only recognized field names:
        #params = {k: v for k, v in params.items() if k in SeriesLoader.BinaryLoadParameters._fields}
        for k in params.keys():
            if not k in SeriesLoader.BinaryLoadParameters._fields:
                del params[k]
        keywordparams = {'nkeys': nkeys, 'nvalues': nvalues, 'keyformat': keytype, 'format': valuetype}
        #keywordparams = {k: v for k, v in keywordparams.items() if v}
        for k, v in keywordparams.items():
            if not v:
                del keywordparams[k]
        params.update(keywordparams)
        return SeriesLoader.BinaryLoadParameters(**params)

    @staticmethod
    def __checkBinaryParametersAreSpecified(paramsObj):
        """Throws ValueError if any of the field values in the passed namedtuple instance evaluate to False.

        Note this is okay only so long as zero is not a valid parameter value. Hmm.
        """
        missing = []
        for paramname, paramval in paramsObj._asdict().iteritems():
            if not paramval:
                missing.append(paramname)
        if missing:
            raise ValueError("Missing parameters to load binary series files - " +
                             "these must be given either as arguments or in a configuration file: " +
                             str(tuple(missing)))

    def fromBinary(self, datafile, ext='bin', conffilename='conf.json',
                   nkeys=None, nvalues=None, keytype=None, valuetype=None):

        datafile = self.__normalizeDatafilePattern(datafile, ext)

        paramsObj = self.__loadParametersAndDefaults(datafile, conffilename, nkeys, nvalues, keytype, valuetype)
        self.__checkBinaryParametersAreSpecified(paramsObj)

        keydtype = dtype(paramsObj.keyformat)
        valdtype = dtype(paramsObj.format)

        keysize = paramsObj.nkeys * keydtype.itemsize
        recordsize = keysize + paramsObj.nvalues * valdtype.itemsize

        lines = self.sc.newAPIHadoopFile(datafile, 'thunder.util.io.hadoop.FixedLengthBinaryInputFormat',
                                              'org.apache.hadoop.io.LongWritable',
                                              'org.apache.hadoop.io.BytesWritable',
                                              conf={'recordLength': str(recordsize)})

        def _parseKeysFromBinaryBuffer(buf, keydtype_, keybufsize):
            return frombuffer(buffer(buf, 0, keybufsize), dtype=keydtype_)

        def _parseValsFromBinaryBuffer(buf, valsdtype_, keybufsize):
            # note this indeed takes *key* buffer size as an argument, not valbufsize
            return frombuffer(buffer(buf, keybufsize), dtype=valsdtype_)

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
        """Returns a dict loaded from a json file

        Looks for file named _conffile_ in same directory as _datafile_.

        Returns {} if file not found
        """
        if not conffile:
            return {}
        reader = getFileReaderForPath(datafile)()
        try:
            jsonbuf = reader.read(datafile, filename=conffile)
        except Exception:
            # depending on the reader, any number of different exceptions
            # could be getting thrown here. Would be good to standardize a bit
            # if possible.
            # todo: this is not always an error we want to squelch :(
            jsonbuf = "{}"
        return json.loads(jsonbuf)



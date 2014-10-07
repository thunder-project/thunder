from collections import namedtuple
import json
from numpy import array, dtype, frombuffer, arange, load
from scipy.io import loadmat
import urlparse
from thunder.rdds.rddio.readers import getFileReaderForPath, FileNotFoundError
from thunder import Series


class SeriesLoader(object):

    def __init__(self, sparkcontext, minPartitions=None):
        self.sc = sparkcontext
        self.minPartitions = minPartitions

    @staticmethod
    def __normalizeDatafilePattern(datapath, ext):
        if ext and (not datapath.endswith(ext)):
            if datapath.endswith("*"):
                datapath += ext
            elif datapath.endswith("/"):
                datapath += "*" + ext
            else:
                datapath += "/*" + ext

        parseresult = urlparse.urlparse(datapath)
        if parseresult.scheme:
            # this appears to already be a fully-qualified URI
            return datapath
        else:
            return "file://" + datapath

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
        """
        Load a Series object from a directory of binary files.

        Parameters
        ----------

        datafile: string URI or local filesystem path
            Specifies the directory in which to look for binary files. All files with the extension given by 'ext' in
            the passed directory will be loaded.

        """

        paramsObj = self.__loadParametersAndDefaults(datafile, conffilename, nkeys, nvalues, keytype, valuetype)
        self.__checkBinaryParametersAreSpecified(paramsObj)

        datafile = self.__normalizeDatafilePattern(datafile, ext)

        keydtype = dtype(paramsObj.keyformat)
        valdtype = dtype(paramsObj.format)

        keysize = paramsObj.nkeys * keydtype.itemsize
        recordsize = keysize + paramsObj.nvalues * valdtype.itemsize

        lines = self.sc.newAPIHadoopFile(datafile, 'thunder.util.io.hadoop.FixedLengthBinaryInputFormat',
                                              'org.apache.hadoop.io.LongWritable',
                                              'org.apache.hadoop.io.BytesWritable',
                                              conf={'recordLength': str(recordsize)})

        data = lines.map(lambda (_, v):
                         (tuple(frombuffer(buffer(v, 0, keysize), dtype=keydtype)),
                          frombuffer(buffer(v, keysize), dtype=valdtype)))

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
        except FileNotFoundError:
            return {}
        return json.loads(jsonbuf)
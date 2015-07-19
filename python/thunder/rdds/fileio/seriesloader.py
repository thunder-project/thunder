"""Provides SeriesLoader object and helpers, used to read Series data from disk or other filesystems.
"""
from collections import namedtuple
import json
from numpy import array, arange, frombuffer, load, ndarray, vstack
from numpy import dtype as dtypeFunc
from scipy.io import loadmat
from cStringIO import StringIO
import itertools
import struct
import urlparse

from thunder.rdds.fileio.writers import getParallelWriterForPath
from thunder.rdds.keys import Dimensions
from thunder.rdds.fileio.readers import getFileReaderForPath, FileNotFoundError, appendExtensionToPathSpec
from thunder.rdds.imgblocks.blocks import SimpleBlocks
from thunder.rdds.series import Series


class SeriesLoader(object):
    """Loader object used to instantiate Series data stored in a variety of formats.
    """
    def __init__(self, sparkContext, minPartitions=None):
        """Initialize a new SeriesLoader object.

        Parameters
        ----------
        sparkcontext: SparkContext
            The pyspark SparkContext object used by the current Thunder environment.

        minPartitions: int
            minimum number of partitions to use when loading data. (Used by fromText, fromMatLocal, and fromNpyLocal)
        """
        from thunder.utils.aws import AWSCredentials
        self.sc = sparkContext
        self.minPartitions = minPartitions
        self.awsCredentialsOverride = AWSCredentials.fromContext(sparkContext)

    def _checkOverwrite(self, outputDirPath):
        from thunder.utils.common import raiseErrorIfPathExists
        raiseErrorIfPathExists(outputDirPath, awsCredentialsOverride=self.awsCredentialsOverride)

    def fromArrays(self, arrays, npartitions=None):
        """
        Create a Series object from a sequence of 1d numpy arrays on the driver.
        """
        # recast singleton
        if isinstance(arrays, ndarray):
            arrays = [arrays]

        # check shape and dtype
        shape = arrays[0].shape
        dtype = arrays[0].dtype
        for ary in arrays:
            if not ary.shape == shape:
                raise ValueError("Inconsistent array shapes: first array had shape %s, but other array has shape %s" %
                                 (str(shape), str(ary.shape)))
            if not ary.dtype == dtype:
                raise ValueError("Inconsistent array dtypes: first array had dtype %s, but other array has dtype %s" %
                                 (str(dtype), str(ary.dtype)))

        # generate linear keys
        keys = map(lambda k: (k,), xrange(0, len(arrays)))

        return Series(self.sc.parallelize(zip(keys, arrays), npartitions), dtype=str(dtype))

    def fromArraysAsImages(self, arrays):
        """Create a Series object from a sequence of numpy ndarrays resident in memory on the driver.

        The arrays will be interpreted as though each represents a single time point - effectively the same
        as if converting Images to a Series, with each array representing a volume image at a particular
        point in time. Thus in the resulting Series, the value of the record with key (0,0,0) will be
        array([arrays[0][0,0,0], arrays[1][0,0,0],... arrays[n][0,0,0]).

        The dimensions of the resulting Series will be *opposite* that of the passed numpy array. Their dtype will not
        be changed.
        """
        # if passed a single array, cast it to a sequence of length 1
        if isinstance(arrays, ndarray):
            arrays = [arrays]

        # check that shapes of passed arrays are consistent
        shape = arrays[0].shape
        dtype = arrays[0].dtype
        for ary in arrays:
            if not ary.shape == shape:
                raise ValueError("Inconsistent array shapes: first array had shape %s, but other array has shape %s" %
                                 (str(shape), str(ary.shape)))
            if not ary.dtype == dtype:
                raise ValueError("Inconsistent array dtypes: first array had dtype %s, but other array has dtype %s" %
                                 (str(dtype), str(ary.dtype)))

        # get indices so that fastest index changes first
        shapeiters = (xrange(n) for n in shape)
        keys = [idx[::-1] for idx in itertools.product(*shapeiters)]

        values = vstack([ary.ravel() for ary in arrays]).T

        dims = Dimensions.fromTuple(shape[::-1])

        return Series(self.sc.parallelize(zip(keys, values), self.minPartitions), dims=dims, dtype=str(dtype))

    @staticmethod
    def __normalizeDatafilePattern(dataPath, ext):
        dataPath = appendExtensionToPathSpec(dataPath, ext)
        # we do need to prepend a scheme here, b/c otherwise the Hadoop based readers
        # will adopt their default behavior and start looking on hdfs://.

        parseResult = urlparse.urlparse(dataPath)
        if parseResult.scheme:
            # this appears to already be a fully-qualified URI
            return dataPath
        else:
            # this looks like a local path spec
            # check whether we look like an absolute or a relative path
            import os
            dirComponent, fileComponent = os.path.split(dataPath)
            if not os.path.isabs(dirComponent):
                # need to make relative local paths absolute; our file scheme parsing isn't all that it could be.
                dirComponent = os.path.abspath(dirComponent)
                dataPath = os.path.join(dirComponent, fileComponent)
            return "file://" + dataPath

    def fromText(self, dataPath, nkeys=None, ext="txt", dtype='float64'):
        """
        Loads Series data from text files.

        Parameters
        ----------
        dataPath : string
            Specifies the file or files to be loaded. dataPath may be either a URI (with scheme specified) or a path
            on the local filesystem.
            If a path is passed (determined by the absence of a scheme component when attempting to parse as a URI),
            and it is not already a wildcard expression and does not end in <ext>, then it will be converted into a
            wildcard pattern by appending '/*.ext'. This conversion can be avoided by passing a "file://" URI.

        dtype: dtype or dtype specifier, default 'float64'

        """
        dataPath = self.__normalizeDatafilePattern(dataPath, ext)

        def parse(line, nkeys_):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys_:], dtype=dtype)
            keys = tuple(int(x) for x in vec[:nkeys_])
            return keys, ts

        lines = self.sc.textFile(dataPath, self.minPartitions)
        data = lines.map(lambda x: parse(x, nkeys))
        return Series(data, dtype=str(dtype))

    # keytype, valuetype here violate camelCasing convention for consistence with JSON conf file format
    BinaryLoadParameters = namedtuple('BinaryLoadParameters', 'nkeys nvalues keytype valuetype')
    BinaryLoadParameters.__new__.__defaults__ = (None, None, 'int16', 'int16')

    def __loadParametersAndDefaults(self, dataPath, confFilename, nkeys, nvalues, keyType, valueType):
        """Collects parameters to use for binary series loading.

        Priority order is as follows:
        1. parameters specified as keyword arguments;
        2. parameters specified in a conf.json file on the local filesystem;
        3. default parameters

        Returns
        -------
        BinaryLoadParameters instance
        """
        params = self.loadConf(dataPath, confFilename=confFilename)

        # filter dict to include only recognized field names:
        for k in params.keys():
            if k not in SeriesLoader.BinaryLoadParameters._fields:
                del params[k]
        keywordParams = {'nkeys': nkeys, 'nvalues': nvalues, 'keytype': keyType, 'valuetype': valueType}
        for k, v in keywordParams.items():
            if not v:
                del keywordParams[k]
        params.update(keywordParams)
        return SeriesLoader.BinaryLoadParameters(**params)

    @staticmethod
    def __checkBinaryParametersAreSpecified(paramsObj):
        """Throws ValueError if any of the field values in the passed namedtuple instance evaluate to False.

        Note this is okay only so long as zero is not a valid parameter value. Hmm.
        """
        missing = []
        for paramName, paramVal in paramsObj._asdict().iteritems():
            if not paramVal:
                missing.append(paramName)
        if missing:
            raise ValueError("Missing parameters to load binary series files - " +
                             "these must be given either as arguments or in a configuration file: " +
                             str(tuple(missing)))

    def fromBinary(self, dataPath, ext='bin', confFilename='conf.json',
                   nkeys=None, nvalues=None, keyType=None, valueType=None,
                   newDtype='smallfloat', casting='safe', maxPartitionSize='32mb'):
        """
        Load a Series object from a directory of binary files.

        Parameters
        ----------

        dataPath : string URI or local filesystem path
            Specifies the directory or files to be loaded. May be formatted as a URI string with scheme (e.g. "file://",
            "s3n://", or "gs://"). If no scheme is present, will be interpreted as a path on the local filesystem. This path
            must be valid on all workers. Datafile may also refer to a single file, or to a range of files specified
            by a glob-style expression using a single wildcard character '*'.

        newDtype : dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
            Numpy dtype of output series data. Most methods expect Series data to be floating-point. Input data will be
            cast to the requested `newdtype` if not None - see Data `astype()` method.

        casting : 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        maxPartitionSize : str, optional, default = '32mb'
            Maximum size of partitions as Java-style memory, will indirectly control the number of partitions

        """

        paramsObj = self.__loadParametersAndDefaults(dataPath, confFilename, nkeys, nvalues, keyType, valueType)
        self.__checkBinaryParametersAreSpecified(paramsObj)

        dataPath = self.__normalizeDatafilePattern(dataPath, ext)

        keyDtype = dtypeFunc(paramsObj.keytype)
        valDtype = dtypeFunc(paramsObj.valuetype)

        keySize = paramsObj.nkeys * keyDtype.itemsize
        recordSize = keySize + paramsObj.nvalues * valDtype.itemsize

        from thunder.utils.common import parseMemoryString
        if isinstance(maxPartitionSize, basestring):
            size = parseMemoryString(maxPartitionSize)
        else:
            raise Exception("Invalid size specification")
        hadoopConf = {'recordLength': str(recordSize), 'mapred.max.split.size': str(size)}

        lines = self.sc.newAPIHadoopFile(dataPath, 'thunder.util.io.hadoop.FixedLengthBinaryInputFormat',
                                         'org.apache.hadoop.io.LongWritable',
                                         'org.apache.hadoop.io.BytesWritable',
                                         conf=hadoopConf)

        data = lines.map(lambda (_, v):
                         (tuple(int(x) for x in frombuffer(buffer(v, 0, keySize), dtype=keyDtype)),
                          frombuffer(buffer(v, keySize), dtype=valDtype)))

        return Series(data, dtype=str(valDtype), index=arange(paramsObj.nvalues)).astype(newDtype, casting)

    def __saveSeriesRdd(self, seriesBlocks, outputDirPath, dims, npointsInSeries, dtype, overwrite=False):
        if not overwrite:
            self._checkOverwrite(outputDirPath)
            overwrite = True  # prevent additional downstream checks for this path
        writer = getParallelWriterForPath(outputDirPath)(outputDirPath, overwrite=overwrite,
                                                         awsCredentialsOverride=self.awsCredentialsOverride)

        def blockToBinarySeries(kvIter):
            label = None
            keyPacker = None
            buf = StringIO()
            for seriesKey, series in kvIter:
                if keyPacker is None:
                    keyPacker = struct.Struct('h'*len(seriesKey))
                    label = SimpleBlocks.getBinarySeriesNameForKey(seriesKey) + ".bin"
                buf.write(keyPacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return [(label, val)]

        seriesBlocks.mapPartitions(blockToBinarySeries).foreach(writer.writerFcn)
        writeSeriesConfig(outputDirPath, len(dims), npointsInSeries, valueType=dtype, overwrite=overwrite,
                          awsCredentialsOverride=self.awsCredentialsOverride)

    def fromMatLocal(self, dataPath, varName, keyFile=None):
        """Loads Series data stored in a Matlab .mat file.

        `datafile` must refer to a path visible to all workers, such as on NFS or similar mounted shared filesystem.
        """
        data = loadmat(dataPath)[varName]
        if data.ndim > 2:
            raise IOError('Input data must be one or two dimensional')
        if keyFile:
            keys = map(lambda x: tuple(x), loadmat(keyFile)['keys'])
        else:
            keys = arange(0, data.shape[0])

        rdd = Series(self.sc.parallelize(zip(keys, data), self.minPartitions), dtype=str(data.dtype))

        return rdd

    def fromNpyLocal(self, dataPath, keyFile=None):
        """Loads Series data stored in the numpy save() .npy format.

        `datafile` must refer to a path visible to all workers, such as on NFS or similar mounted shared filesystem.
        """
        data = load(dataPath)
        if data.ndim > 2:
            raise IOError('Input data must be one or two dimensional')
        if keyFile:
            keys = map(lambda x: tuple(x), load(keyFile))
        else:
            keys = arange(0, data.shape[0])

        rdd = Series(self.sc.parallelize(zip(keys, data), self.minPartitions), dtype=str(data.dtype))

        return rdd

    def loadConf(self, dataPath, confFilename='conf.json'):
        """Returns a dict loaded from a json file.

        Looks for file named `conffile` in same directory as `dataPath`

        Returns {} if file not found
        """
        if not confFilename:
            return {}

        reader = getFileReaderForPath(dataPath)(awsCredentialsOverride=self.awsCredentialsOverride)
        try:
            jsonBuf = reader.read(dataPath, filename=confFilename)
        except FileNotFoundError:
            return {}

        params = json.loads(jsonBuf)

        if 'format' in params:
            raise Exception("Numerical format of value should be specified as 'valuetype', not 'format'")
        if 'keyformat' in params:
            raise Exception("Numerical format of key should be specified as 'keytype', not 'keyformat'")

        return params


def writeSeriesConfig(outputDirPath, nkeys, nvalues, keyType='int16', valueType='int16',
                      confFilename="conf.json", overwrite=True, awsCredentialsOverride=None):
    """
    Helper function to write out a conf.json file with required information to load Series binary data.
    """
    import json
    from thunder.rdds.fileio.writers import getFileWriterForPath

    filewriterClass = getFileWriterForPath(outputDirPath)
    # write configuration file
    # config JSON keys are lowercased "valuetype", "keytype", not valueType, keyType
    conf = {'input': outputDirPath,
            'nkeys': nkeys, 'nvalues': nvalues,
            'valuetype': str(valueType), 'keytype': str(keyType)}

    confWriter = filewriterClass(outputDirPath, confFilename, overwrite=overwrite,
                                 awsCredentialsOverride=awsCredentialsOverride)
    confWriter.writeFile(json.dumps(conf, indent=2))

    # touch "SUCCESS" file as final action
    successWriter = filewriterClass(outputDirPath, "SUCCESS", overwrite=overwrite,
                                    awsCredentialsOverride=awsCredentialsOverride)
    successWriter.writeFile('')


from collections import namedtuple
import json
from numpy import array, dtype, frombuffer, arange, load, vstack, unravel_index
from scipy.io import loadmat
from cStringIO import StringIO
import struct
import urlparse
import math

from thunder.rdds.data import parseMemoryString
from thunder.rdds.fileio.writers import getParallelWriterForPath
from thunder.rdds.imageblocks import ImageBlocks
from thunder.rdds.fileio.readers import getFileReaderForPath, FileNotFoundError, selectByStartAndStopIndices
from thunder.rdds.series import Series


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
            Specifies the directory or files to be loaded. May be formatted as a URI string with scheme (e.g. "file://",
            "s3n://". If no scheme is present, will be interpreted as a path on the local filesystem. This path
            must be valid on all workers. Datafile may also refer to a single file, or to a range of files specified
            by a glob-style expression using a single wildcard character '*'.

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
                         (tuple(int(x) for x in frombuffer(buffer(v, 0, keysize), dtype=keydtype)),
                          frombuffer(buffer(v, keysize), dtype=valdtype)))

        return Series(data)

    def _getSeriesBlocksFromStack(self, datapath, dims, ext="stack", blockSize="150M", datatype='int16',
                                  startidx=None, stopidx=None):
        """Create an RDD of <string blocklabel, (int k-tuple indices,

        Parameters
        ----------

        datafile: string URI or local filesystem path
            Specifies the directory or files to be loaded. May be formatted as a URI string with scheme (e.g. "file://",
            "s3n://". If no scheme is present, will be interpreted as a path on the local filesystem. This path
            must be valid on all workers. Datafile may also refer to a single file, or to a range of files specified
            by a glob-style expression using a single wildcard character '*'.


        Returns
        ---------
        pair of (RDD, ntimepoints)

        RDD: sequence of keys, values pairs
            (call using flatMap)

        RDD Key: tuple of int
            zero-based indicies of position within original image volume

        RDD Value: numpy array of datatype
            series of values at position across loaded image volumes

        ntimepoints: int
            number of time points in returned series, determined from number of stack files found at datapath

        """

        datapath = self.__normalizeDatafilePattern(datapath, ext)
        blockSize = parseMemoryString(blockSize)
        totaldim = reduce(lambda x, y: x*y, dims)
        datatype = dtype(datatype)

        reader = getFileReaderForPath(datapath)()
        filenames = reader.list(datapath)
        if not filenames:
            raise IOError("No files found for path '%s'" % datapath)
        filenames = selectByStartAndStopIndices(filenames, startidx, stopidx)

        datasize = totaldim * len(filenames) * datatype.itemsize
        nblocks = max(datasize / blockSize, 1)  # integer division

        if len(dims) >= 3:
            # for 3D stacks, do calculations to ensure that
            # different planes appear in distinct files
            blocksperplane = max(nblocks / dims[-1], 1)

            pixperplane = reduce(lambda x, y: x*y, dims[:-1])

            # get the greatest number of blocks in a plane (up to as many as requested) that still divide the plane
            # evenly. This will always be at least one.
            kupdated = [x for x in range(1, blocksperplane+1) if not pixperplane % x][-1]
            nblocks = kupdated * dims[-1]
            blockSizePerStack = (totaldim / nblocks) * datatype.itemsize
        else:
            # otherwise just round to make contents divide into nearly even blocks
            blockSizePerStack = int(math.ceil(totaldim / float(nblocks)))
            nblocks = int(math.ceil(totaldim / float(blockSizePerStack)))
            blockSizePerStack *= datatype.itemsize

        filesize = totaldim * datatype.itemsize

        def readblock(blocknum):
            # copy size out from closure; will modify later:
            blockSizePerStack_ = blockSizePerStack
            # get start position for this block
            position = blocknum * blockSizePerStack_

            # adjust if at end of file
            if (position + blockSizePerStack_) > filesize:
                blockSizePerStack_ = int(filesize - position)
            # loop over files, loading one block from each
            bufs = []

            for fname in filenames:
                buf = reader.read(fname, startOffset=position, size=blockSizePerStack_)
                bufs.append(frombuffer(buf, dtype=datatype))

            buf = vstack(bufs).T  # dimensions are now linindex x time (images)
            del bufs

            # append subscript keys based on dimensions
            itemposition = position / datatype.itemsize
            itemblocksize = blockSizePerStack_ / datatype.itemsize
            linindx = arange(itemposition, itemposition + itemblocksize)  # zero-based

            keys = zip(*map(tuple, unravel_index(linindx, dims, order='F')))
            return zip(keys, buf)

        # map over blocks
        return self.sc.parallelize(range(0, nblocks), nblocks).flatMap(lambda bn: readblock(bn)), len(filenames)

    def fromStack(self, datapath, dims, ext="stack", blockSize="150M", datatype='int16', startidx=None, stopidx=None):
        """Load a Series object directly from binary image stack files.

        Parameters
        ----------

        datafile: string URI or local filesystem path
            Specifies the directory or files to be loaded. May be formatted as a URI string with scheme (e.g. "file://",
            "s3n://". If no scheme is present, will be interpreted as a path on the local filesystem. This path
            must be valid on all workers. Datafile may also refer to a single file, or to a range of files specified
            by a glob-style expression using a single wildcard character '*'.
        """
        seriesblocks, npointsinseries = self._getSeriesBlocksFromStack(datapath, dims, ext=ext, blockSize=blockSize,
                                                                       datatype=datatype, startidx=startidx,
                                                                       stopidx=stopidx)
        # TODO: initialize index here?
        return Series(seriesblocks, dims=dims)

    def saveFromStack(self, datapath, outputdirname, dims, ext="stack", blockSize="150M", datatype='int16',
                      startidx=None, stopidx=None, overwrite=False):

        writer = getParallelWriterForPath(outputdirname)(outputdirname, overwrite=overwrite)
        seriesblocks, npointsinseries = self._getSeriesBlocksFromStack(datapath, dims, ext=ext, blockSize=blockSize,
                                                                       datatype=datatype, startidx=startidx,
                                                                       stopidx=stopidx)

        def blockToBinarySeries(kviter):
            label = None
            keypacker = None
            buf = StringIO()
            for seriesKey, series in kviter:
                if keypacker is None:
                    keypacker = struct.Struct('h'*len(seriesKey))
                    label = ImageBlocks.getBinarySeriesNameForKey(seriesKey) + ".bin"
                buf.write(keypacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return [(label, val)]

        seriesblocks.mapPartitions(blockToBinarySeries).foreach(writer.writerFcn)
        writeSeriesConfig(outputdirname, len(dims), npointsinseries, dims=dims, valuetype=datatype,
                          overwrite=overwrite)

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


def writeSeriesConfig(outputdirname, nkeys, nvalues, dims=None, keytype='int16', valuetype='int16', confname="conf.json",
                      overwrite=True):

    import json
    from thunder.rdds.fileio.writers import getFileWriterForPath

    filewriterclass = getFileWriterForPath(outputdirname)
    # write configuration file
    conf = {'input': outputdirname,
            'nkeys': nkeys, 'nvalues': nvalues,
            'format': str(valuetype), 'keyformat': str(keytype)}
    if dims:
        conf["dims"] = dims

    confwriter = filewriterclass(outputdirname, confname, overwrite=overwrite)
    confwriter.writeFile(json.dumps(conf, indent=2))

    # touch "SUCCESS" file as final action
    successwriter = filewriterclass(outputdirname, "SUCCESS", overwrite=overwrite)
    successwriter.writeFile('')

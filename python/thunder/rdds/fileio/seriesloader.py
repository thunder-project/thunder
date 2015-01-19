"""Provides SeriesLoader object and helpers, used to read Series data from disk or other filesystems.
"""
from collections import namedtuple
import json
from numpy import array, arange, frombuffer, load, ndarray, unravel_index, vstack
from numpy import dtype as dtypefunc
from scipy.io import loadmat
from cStringIO import StringIO
import itertools
import struct
import urlparse
import math

from thunder.rdds.fileio.writers import getParallelWriterForPath
from thunder.rdds.keys import Dimensions
from thunder.rdds.fileio.readers import getFileReaderForPath, FileNotFoundError, selectByStartAndStopIndices, \
    appendExtensionToPathSpec
from thunder.rdds.imgblocks.blocks import SimpleBlocks
from thunder.rdds.series import Series
from thunder.utils.common import parseMemoryString, smallest_float_type


class SeriesLoader(object):
    """Loader object used to instantiate Series data stored in a variety of formats.
    """
    def __init__(self, sparkcontext, minPartitions=None):
        """Initialize a new SeriesLoader object.

        Parameters
        ----------
        sparkcontext: SparkContext
            The pyspark SparkContext object used by the current Thunder environment.

        minPartitions: int
            minimum number of partitions to use when loading data. (Used by fromText, fromMatLocal, and fromNpyLocal)
        """
        self.sc = sparkcontext
        self.minPartitions = minPartitions

    def fromArrays(self, arrays):
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
    def __normalizeDatafilePattern(datapath, ext):
        datapath = appendExtensionToPathSpec(datapath, ext)
        # we do need to prepend a scheme here, b/c otherwise the Hadoop based readers
        # will adopt their default behavior and start looking on hdfs://.

        parseresult = urlparse.urlparse(datapath)
        if parseresult.scheme:
            # this appears to already be a fully-qualified URI
            return datapath
        else:
            # this looks like a local path spec
            # check whether we look like an absolute or a relative path
            import os
            dircomponent, filecomponent = os.path.split(datapath)
            if not os.path.isabs(dircomponent):
                # need to make relative local paths absolute; our file scheme parsing isn't all that it could be.
                dircomponent = os.path.abspath(dircomponent)
                datapath = os.path.join(dircomponent, filecomponent)
            return "file://" + datapath

    def fromText(self, datafile, nkeys=None, ext="txt", dtype='float64'):
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

        dtype: dtype or dtype specifier, default 'float64'

        """
        datafile = self.__normalizeDatafilePattern(datafile, ext)

        def parse(line, nkeys_):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys_:], dtype=dtype)
            keys = tuple(int(x) for x in vec[:nkeys_])
            return keys, ts

        lines = self.sc.textFile(datafile, self.minPartitions)
        data = lines.map(lambda x: parse(x, nkeys))
        return Series(data, dtype=str(dtype))

    BinaryLoadParameters = namedtuple('BinaryLoadParameters', 'nkeys nvalues keytype valuetype')
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
        for k in params.keys():
            if not k in SeriesLoader.BinaryLoadParameters._fields:
                del params[k]
        keywordparams = {'nkeys': nkeys, 'nvalues': nvalues, 'keytype': keytype, 'valuetype': valuetype}
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
                   nkeys=None, nvalues=None, keytype=None, valuetype=None,
                   newdtype='smallfloat', casting='safe'):
        """
        Load a Series object from a directory of binary files.

        Parameters
        ----------

        datafile: string URI or local filesystem path
            Specifies the directory or files to be loaded. May be formatted as a URI string with scheme (e.g. "file://",
            "s3n://". If no scheme is present, will be interpreted as a path on the local filesystem. This path
            must be valid on all workers. Datafile may also refer to a single file, or to a range of files specified
            by a glob-style expression using a single wildcard character '*'.

        newdtype: dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
            Numpy dtype of output series data. Most methods expect Series data to be floating-point. Input data will be
            cast to the requested `newdtype` if not None - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        """

        paramsObj = self.__loadParametersAndDefaults(datafile, conffilename, nkeys, nvalues, keytype, valuetype)
        self.__checkBinaryParametersAreSpecified(paramsObj)

        datafile = self.__normalizeDatafilePattern(datafile, ext)

        keydtype = dtypefunc(paramsObj.keytype)
        valdtype = dtypefunc(paramsObj.valuetype)

        keysize = paramsObj.nkeys * keydtype.itemsize
        recordsize = keysize + paramsObj.nvalues * valdtype.itemsize

        lines = self.sc.newAPIHadoopFile(datafile, 'thunder.util.io.hadoop.FixedLengthBinaryInputFormat',
                                         'org.apache.hadoop.io.LongWritable',
                                         'org.apache.hadoop.io.BytesWritable',
                                         conf={'recordLength': str(recordsize)})

        data = lines.map(lambda (_, v):
                         (tuple(int(x) for x in frombuffer(buffer(v, 0, keysize), dtype=keydtype)),
                          frombuffer(buffer(v, keysize), dtype=valdtype)))

        return Series(data, dtype=str(valdtype), index=arange(paramsObj.nvalues)).astype(newdtype, casting)

    def _getSeriesBlocksFromStack(self, datapath, dims, ext="stack", blockSize="150M", datatype='int16',
                                  newdtype='smallfloat', casting='safe', startidx=None, stopidx=None, recursive=False):
        """Create an RDD of <string blocklabel, (int k-tuple indices, array of datatype values)>

        Parameters
        ----------

        datafile: string URI or local filesystem path
            Specifies the directory or files to be loaded. May be formatted as a URI string with scheme (e.g. "file://",
            "s3n://". If no scheme is present, will be interpreted as a path on the local filesystem. This path
            must be valid on all workers. Datafile may also refer to a single file, or to a range of files specified
            by a glob-style expression using a single wildcard character '*'.

        dims: tuple of positive int
            Dimensions of input image data, ordered with the fastest-changing dimension first.

        datatype: dtype or dtype specifier, optional, default 'int16'
            Numpy dtype of input stack data

        newdtype: floating-point dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
            Numpy dtype of output series data. Series data must be floating-point. Input data will be cast to the
            requested `newdtype` - see numpy `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        recursive: boolean, default False
            If true, will recursively descend directories rooted at datapath, loading all files in the tree that
            have an extension matching 'ext'. Recursive loading is currently only implemented for local filesystems
            (not s3).

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

        newdtype: string
            string representation of numpy data type of returned blocks

        """
        datapath = self.__normalizeDatafilePattern(datapath, ext)
        blockSize = parseMemoryString(blockSize)
        totaldim = reduce(lambda x_, y_: x_*y_, dims)
        datatype = dtypefunc(datatype)
        if newdtype is None or newdtype == '':
            newdtype = str(datatype)
        elif newdtype == 'smallfloat':
            newdtype = str(smallest_float_type(datatype))
        else:
            newdtype = str(newdtype)

        reader = getFileReaderForPath(datapath)()
        filenames = reader.list(datapath, startidx=startidx, stopidx=stopidx, recursive=recursive)
        if not filenames:
            raise IOError("No files found for path '%s'" % datapath)

        datasize = totaldim * len(filenames) * datatype.itemsize
        nblocks = max(datasize / blockSize, 1)  # integer division

        if len(dims) >= 3:
            # for 3D stacks, do calculations to ensure that
            # different planes appear in distinct files
            blocksperplane = max(nblocks / dims[-1], 1)

            pixperplane = reduce(lambda x_, y_: x_*y_, dims[:-1])  # all but last dimension

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
            buf = buf.astype(newdtype, casting=casting, copy=False)

            # append subscript keys based on dimensions
            itemposition = position / datatype.itemsize
            itemblocksize = blockSizePerStack_ / datatype.itemsize
            linindx = arange(itemposition, itemposition + itemblocksize)  # zero-based

            keys = zip(*map(tuple, unravel_index(linindx, dims, order='F')))
            return zip(keys, buf)

        # map over blocks
        return (self.sc.parallelize(range(0, nblocks), nblocks).flatMap(lambda bn: readblock(bn)),
                len(filenames), newdtype)

    @staticmethod
    def __readMetadataFromFirstPageOfMultiTif(reader, filepath):
        import thunder.rdds.fileio.multitif as multitif

        # read first page of first file to get expected image size
        tiffp = reader.open(filepath)
        tiffparser = multitif.TiffParser(tiffp, debug=False)
        tiffheaders = multitif.TiffData()
        tiffparser.parseFileHeader(destination_tiff=tiffheaders)
        firstifd = tiffparser.parseNextImageFileDirectory(destination_tiff=tiffheaders)
        if not firstifd.isLuminanceImage():
            raise ValueError(("File %s does not appear to be a luminance " % filepath) +
                             "(greyscale or bilevel) TIF image, " +
                             "which are the only types currently supported")

        # keep reading pages until we reach the end of the file, in order to get number of planes:
        while tiffparser.parseNextImageFileDirectory(destination_tiff=tiffheaders):
            pass

        # get dimensions
        npages = len(tiffheaders.ifds)
        height = firstifd.getImageHeight()
        width = firstifd.getImageWidth()

        # get datatype
        bitspersample = firstifd.getBitsPerSample()
        if not (bitspersample in (8, 16, 32, 64)):
            raise ValueError("Only 8, 16, 32, or 64 bit per pixel TIF images are supported, got %d" % bitspersample)

        sampleformat = firstifd.getSampleFormat()
        if sampleformat == multitif.SAMPLE_FORMAT_UINT:
            dtstr = 'uint'
        elif sampleformat == multitif.SAMPLE_FORMAT_INT:
            dtstr = 'int'
        elif sampleformat == multitif.SAMPLE_FORMAT_FLOAT:
            dtstr = 'float'
        else:
            raise ValueError("Unknown TIF SampleFormat tag value %d, should be 1, 2, or 3 for uint, int, or float"
                             % sampleformat)
        datatype = dtstr+str(bitspersample)

        return height, width, npages, datatype

    def _getSeriesBlocksFromMultiTif(self, datapath, ext="tif", blockSize="150M",
                                     newdtype='smallfloat', casting='safe', startidx=None, stopidx=None,
                                     recursive=False):
        import thunder.rdds.fileio.multitif as multitif
        import itertools
        from PIL import Image
        import io

        datapath = self.__normalizeDatafilePattern(datapath, ext)
        blockSize = parseMemoryString(blockSize)

        reader = getFileReaderForPath(datapath)()
        filenames = reader.list(datapath, startidx=startidx, stopidx=stopidx, recursive=recursive)
        if not filenames:
            raise IOError("No files found for path '%s'" % datapath)
        ntimepoints = len(filenames)

        minimize_reads = datapath.lower().startswith("s3")
        # check PIL version to see whether it is actually pillow or indeed old PIL and choose
        # conversion function appropriately. See ImagesLoader.fromMultipageTif and common.pil_to_array
        # for more explanation.
        isPillow = hasattr(Image, "PILLOW_VERSION")
        if isPillow:
            conversionFcn = array  # use numpy's array() function
        else:
            from thunder.utils.common import pil_to_array
            conversionFcn = pil_to_array  # use our modified version of matplotlib's pil_to_array

        height, width, npages, datatype = SeriesLoader.__readMetadataFromFirstPageOfMultiTif(reader, filenames[0])
        pixelbytesize = dtypefunc(datatype).itemsize
        if newdtype is None or str(newdtype) == '':
            newdtype = str(datatype)
        elif newdtype == 'smallfloat':
            newdtype = str(smallest_float_type(datatype))
        else:
            newdtype = str(newdtype)

        # intialize at one block per plane
        bytesperplane = height * width * pixelbytesize * ntimepoints
        bytesperblock = bytesperplane
        blocksperplane = 1
        # keep dividing while cutting our size in half still leaves us bigger than the requested size
        # should end up no more than 2x blockSize.
        while bytesperblock >= blockSize * 2:
            bytesperblock /= 2
            blocksperplane *= 2

        blocklenPixels = max((height * width) / blocksperplane, 1)  # integer division
        while blocksperplane * blocklenPixels < height * width:  # make sure we're reading the plane fully
            blocksperplane += 1

        # keys will be planeidx, blockidx:
        keys = list(itertools.product(xrange(npages), xrange(blocksperplane)))

        def readblockfromtif(pidxbidx_):
            planeidx, blockidx = pidxbidx_
            blocks = []
            planeshape = None
            blockstart = None
            blockend = None
            for fname in filenames:
                reader_ = getFileReaderForPath(fname)()
                fp = reader_.open(fname)
                try:
                    if minimize_reads:
                        # use multitif module to generate a fake, in-memory one-page tif file
                        # the advantage of this is that it cuts way down on the many small reads
                        # that PIL/pillow will make otherwise, which would be a problem for s3
                        tiffparser_ = multitif.TiffParser(fp, debug=False)
                        tiffilebuffer = multitif.packSinglePage(tiffparser_, page_idx=planeidx)
                        bytebuf = io.BytesIO(tiffilebuffer)
                        try:
                            pilimg = Image.open(bytebuf)
                            ary = conversionFcn(pilimg).T
                        finally:
                            bytebuf.close()
                        del tiffilebuffer, tiffparser_, pilimg, bytebuf
                    else:
                        # read tif using PIL directly
                        pilimg = Image.open(fp)
                        pilimg.seek(planeidx)
                        ary = conversionFcn(pilimg).T
                        del pilimg

                    if not planeshape:
                        planeshape = ary.shape[:]
                        blockstart = blockidx * blocklenPixels
                        blockend = min(blockstart+blocklenPixels, planeshape[0]*planeshape[1])
                    blocks.append(ary.ravel(order='C')[blockstart:blockend])
                    del ary
                finally:
                    fp.close()

            buf = vstack(blocks).T  # dimensions are now linindex x time (images)
            del blocks
            buf = buf.astype(newdtype, casting=casting, copy=False)

            # append subscript keys based on dimensions
            linindx = arange(blockstart, blockend)  # zero-based

            serieskeys = zip(*map(tuple, unravel_index(linindx, planeshape, order='C')))
            # add plane index to end of keys
            serieskeys = [tuple(list(keys_)[::-1]+[planeidx]) for keys_ in serieskeys]
            return zip(serieskeys, buf)

        # map over blocks
        rdd = self.sc.parallelize(keys, len(keys)).flatMap(readblockfromtif)
        dims = (npages, width, height)

        metadata = (dims, ntimepoints, newdtype)
        return rdd, metadata

    def fromStack(self, datapath, dims, ext="stack", blockSize="150M", datatype='int16',
                  newdtype='smallfloat', casting='safe', startidx=None, stopidx=None, recursive=False):
        """Load a Series object directly from binary image stack files.

        Parameters
        ----------

        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename.

        dims: tuple of positive int
            Dimensions of input image data, ordered with the fastest-changing dimension first.

        ext: string, optional, default "stack"
            Extension required on data files to be loaded.

        blocksize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of Series partitions in bytes (or kilobytes, megabytes, gigabytes).

        datatype: dtype or dtype specifier, optional, default 'int16'
            Numpy dtype of input stack data

        newdtype: dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
            Numpy dtype of output series data. Most methods expect Series data to be floating-point. Input data will be
            cast to the requested `newdtype` if not None - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        startidx, stopidx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `datapath` and `ext`. Interpreted according to python slice indexing conventions.

        recursive: boolean, default False
            If true, will recursively descend directories rooted at datapath, loading all files in the tree that
            have an extension matching 'ext'. Recursive loading is currently only implemented for local filesystems
            (not s3).
        """
        seriesblocks, npointsinseries, newdtype = \
            self._getSeriesBlocksFromStack(datapath, dims, ext=ext, blockSize=blockSize, datatype=datatype,
                                           newdtype=newdtype, casting=casting, startidx=startidx, stopidx=stopidx,
                                           recursive=recursive)

        return Series(seriesblocks, dims=dims, dtype=newdtype, index=arange(npointsinseries))

    def fromMultipageTif(self, datapath, ext="tif", blockSize="150M",
                         newdtype='smallfloat', casting='safe',
                         startidx=None, stopidx=None, recursive=False):
        """Load a Series object from multipage tiff files.

        Parameters
        ----------

        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename.

        ext: string, optional, default "tif"
            Extension required on data files to be loaded.

        blocksize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of Series partitions in bytes (or kilobytes, megabytes, gigabytes).

        newdtype: dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
            Numpy dtype of output series data. Most methods expect Series data to be floating-point. Input data will be
            cast to the requested `newdtype` if not None - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        startidx, stopidx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `datapath` and `ext`. Interpreted according to python slice indexing conventions.

        recursive: boolean, default False
            If true, will recursively descend directories rooted at datapath, loading all files in the tree that
            have an extension matching 'ext'. Recursive loading is currently only implemented for local filesystems
            (not s3).
        """
        seriesblocks, metadata = self._getSeriesBlocksFromMultiTif(datapath, ext=ext, blockSize=blockSize,
                                                                   newdtype=newdtype, casting=casting,
                                                                   startidx=startidx, stopidx=stopidx,
                                                                   recursive=False)
        dims, npointsinseries, datatype = metadata
        return Series(seriesblocks, dims=Dimensions.fromTuple(dims[::-1]), dtype=datatype,
                      index=arange(npointsinseries))

    @staticmethod
    def __saveSeriesRdd(seriesblocks, outputdirname, dims, npointsinseries, datatype, overwrite=False):
        if not overwrite:
            from thunder.utils.common import raiseErrorIfPathExists
            raiseErrorIfPathExists(outputdirname)
            overwrite = True  # prevent additional downstream checks for this path

        writer = getParallelWriterForPath(outputdirname)(outputdirname, overwrite=overwrite)

        def blockToBinarySeries(kviter):
            label = None
            keypacker = None
            buf = StringIO()
            for seriesKey, series in kviter:
                if keypacker is None:
                    keypacker = struct.Struct('h'*len(seriesKey))
                    label = SimpleBlocks.getBinarySeriesNameForKey(seriesKey) + ".bin"
                buf.write(keypacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return [(label, val)]

        seriesblocks.mapPartitions(blockToBinarySeries).foreach(writer.writerFcn)
        writeSeriesConfig(outputdirname, len(dims), npointsinseries, valuetype=datatype, overwrite=overwrite)

    def saveFromStack(self, datapath, outputdirpath, dims, ext="stack", blockSize="150M", datatype='int16',
                      newdtype=None, casting='safe', startidx=None, stopidx=None, overwrite=False, recursive=False):
        """Write out data from binary image stack files in the Series data flat binary format.

        Parameters
        ----------
        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename.

        outputdirpath: string
            Path to a directory into which to write Series file output. An outputdir argument may be either a path
            on the local file system or a URI-like format, as in datapath.

        dims: tuple of positive int
            Dimensions of input image data, ordered with the fastest-changing dimension first.

        ext: string, optional, default "stack"
            Extension required on data files to be loaded.

        blocksize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of Series partitions in bytes (or kilobytes, megabytes, gigabytes).

        datatype: dtype or dtype specifier, optional, default 'int16'
            Numpy dtype of input stack data

        newdtype: floating-point dtype or dtype specifier or string 'smallfloat' or None, optional, default None
            Numpy dtype of output series binary data. Input data will be cast to the requested `newdtype` if not None
            - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        startidx, stopidx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `datapath` and `ext`. Interpreted according to python slice indexing conventions.

        overwrite: boolean, optional, default False
            If true, the directory specified by outputdirpath will first be deleted, along with all its contents, if it
            already exists. If false, a ValueError will be thrown if outputdirpath is found to already exist.

        """
        if not overwrite:
            from thunder.utils.common import raiseErrorIfPathExists
            raiseErrorIfPathExists(outputdirpath)
            overwrite = True  # prevent additional downstream checks for this path

        seriesblocks, npointsinseries, newdtype = \
            self._getSeriesBlocksFromStack(datapath, dims, ext=ext, blockSize=blockSize, datatype=datatype,
                                           newdtype=newdtype, casting=casting, startidx=startidx, stopidx=stopidx,
                                           recursive=recursive)

        SeriesLoader.__saveSeriesRdd(seriesblocks, outputdirpath, dims, npointsinseries, newdtype, overwrite=overwrite)

    def saveFromMultipageTif(self, datapath, outputdirpath, ext="tif", blockSize="150M",
                             newdtype=None, casting='safe',
                             startidx=None, stopidx=None, overwrite=False, recursive=False):
        """Write out data from multipage tif files in the Series data flat binary format.

        Parameters
        ----------
        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename.

        outputdirpath: string
            Path to a directory into which to write Series file output. An outputdir argument may be either a path
            on the local file system or a URI-like format, as in datapath.

        ext: string, optional, default "stack"
            Extension required on data files to be loaded.

        blocksize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of Series partitions in bytes (or kilobytes, megabytes, gigabytes).

        newdtype: floating-point dtype or dtype specifier or string 'smallfloat' or None, optional, default None
            Numpy dtype of output series binary data. Input data will be cast to the requested `newdtype` if not None
            - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        startidx, stopidx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `datapath` and `ext`. Interpreted according to python slice indexing conventions.

        overwrite: boolean, optional, default False
            If true, the directory specified by outputdirpath will first be deleted, along with all its contents, if it
            already exists. If false, a ValueError will be thrown if outputdirpath is found to already exist.

        """
        if not overwrite:
            from thunder.utils.common import raiseErrorIfPathExists
            raiseErrorIfPathExists(outputdirpath)
            overwrite = True  # prevent additional downstream checks for this path

        seriesblocks, metadata = self._getSeriesBlocksFromMultiTif(datapath, ext=ext, blockSize=blockSize,
                                                                   newdtype=newdtype, casting=casting,
                                                                   startidx=startidx, stopidx=stopidx,
                                                                   recursive=recursive)
        dims, npointsinseries, datatype = metadata
        SeriesLoader.__saveSeriesRdd(seriesblocks, outputdirpath, dims, npointsinseries, datatype, overwrite=overwrite)

    def fromMatLocal(self, datafile, varname, keyfile=None):
        """Loads Series data stored in a Matlab .mat file.

        `datafile` must refer to a path visible to all workers, such as on NFS or similar mounted shared filesystem.
        """
        data = loadmat(datafile)[varname]
        if data.ndim > 2:
            raise IOError('Input data must be one or two dimensional')
        if keyfile:
            keys = map(lambda x: tuple(x), loadmat(keyfile)['keys'])
        else:
            keys = arange(0, data.shape[0])

        rdd = Series(self.sc.parallelize(zip(keys, data), self.minPartitions), dtype=str(data.dtype))

        return rdd

    def fromNpyLocal(self, datafile, keyfile=None):
        """Loads Series data stored in the numpy save() .npy format.

        `datafile` must refer to a path visible to all workers, such as on NFS or similar mounted shared filesystem.
        """
        data = load(datafile)
        if data.ndim > 2:
            raise IOError('Input data must be one or two dimensional')
        if keyfile:
            keys = map(lambda x: tuple(x), load(keyfile))
        else:
            keys = arange(0, data.shape[0])

        rdd = Series(self.sc.parallelize(zip(keys, data), self.minPartitions), dtype=str(data.dtype))

        return rdd

    @staticmethod
    def loadConf(datafile, conffile='conf.json'):
        """Returns a dict loaded from a json file.

        Looks for file named `conffile` in same directory as `datafile`

        Returns {} if file not found
        """
        if not conffile:
            return {}

        reader = getFileReaderForPath(datafile)()
        try:
            jsonbuf = reader.read(datafile, filename=conffile)
        except FileNotFoundError:
            return {}

        params = json.loads(jsonbuf)

        if 'format' in params:
            raise Exception("Numerical format of value should be specified as 'valuetype', not 'format'")
        if 'keyformat' in params:
            raise Exception("Numerical format of key should be specified as 'keytype', not 'keyformat'")

        return params


def writeSeriesConfig(outputdirname, nkeys, nvalues, keytype='int16', valuetype='int16', confname="conf.json",
                      overwrite=True):
    """Helper function to write out a conf.json file with required information to load Series binary data.
    """
    import json
    from thunder.rdds.fileio.writers import getFileWriterForPath

    filewriterclass = getFileWriterForPath(outputdirname)
    # write configuration file
    conf = {'input': outputdirname,
            'nkeys': nkeys, 'nvalues': nvalues,
            'valuetype': str(valuetype), 'keytype': str(keytype)}

    confwriter = filewriterclass(outputdirname, confname, overwrite=overwrite)
    confwriter.writeFile(json.dumps(conf, indent=2))

    # touch "SUCCESS" file as final action
    successwriter = filewriterclass(outputdirname, "SUCCESS", overwrite=overwrite)
    successwriter.writeFile('')

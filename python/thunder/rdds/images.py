import itertools
import json
import struct
import cStringIO as StringIO
from numpy import ndarray, arange, frombuffer, prod, amax, amin, size, squeeze, reshape, zeros, \
    dtype, dstack
from io import BytesIO
from matplotlib.pyplot import imread, imsave
from series import Series
from thunder.rdds import Data
from thunder.rdds.data import parseMemoryString
from thunder.rdds.readers import getParallelReaderForPath
from thunder.rdds.writers import getFileWriterForPath, getParallelWriterForPath, getCollectedFileWriterForPath


class Images(Data):
    """
    Distributed collection of images or volumes.

    Backed by an RDD of key-value pairs, where the key
    is an identifier and the value is a two or three-dimensional array.
    """

    _metadata = ['_dims', '_nimages', '_dtype']

    def __init__(self, rdd, dims=None, nimages=None, dtype=None):
        super(Images, self).__init__(rdd)
        # todo: add parameter checking here?
        self._dims = dims
        self._nimages = nimages
        self._dtype = str(dtype) if dtype else None

    @property
    def dims(self):
        if self._dims is None:
            self._populateParamsFromFirstRecord()
        return self._dims

    @property
    def nimages(self):
        if self._nimages is None:
            self._nimages = self.rdd.count()
        return self._nimages

    @property
    def dtype(self):
        if not self._dtype:
            self._populateParamsFromFirstRecord()
        return self._dtype

    @property
    def _constructor(self):
        return Images

    def _populateParamsFromFirstRecord(self):
        record = self.rdd.first()
        self._dims = record[1].shape
        self._dtype = str(record[1].dtype)

    @staticmethod
    def _check_type(record):
        if not isinstance(record[0], tuple):
            raise Exception('Keys must be tuples')
        if not isinstance(record[1], ndarray):
            raise Exception('Values must be ndarrays')

    def _toBlocksByImagePlanes(self, groupingDim=-1):
        """Splits Images into ImageBlocks by extracting image planes along specified dimension

        Given an Images data object created from n image files each of dimensions x,y,z (fortran-order),
        this method (with default arguments) will return a new ImageBlocks with n*z items, one for each
        z-plane in the passed images. There will be z unique keys, (0, 0, 0), (0, 0, 1), ... (0, 0, z-1).
        Each value will be an instance of ImageBlockValue, representing an n,z plane within a larger
        volume of dimensions n,x,y,z.

        This method is not expected to be called directly by end users.

        Parameters:
        -----------
        groupingdim: integer, -ndim <= groupingdim < ndim, where ndim is the dimensionality of the image
            Specifies the index of the dimension along which the images are to be divided into planes.
            Negative groupingdims are interpreted as counting from the end of the sequence of dimensions,
            so groupingdim == -1 represents slicing along the last dimension. -1 is the default, corresponding
            to grouping along z-planes for image files with dimensions x, y, z.

        """
        dims = self.dims
        ndim = len(self.dims)
        blocksperdim = [1] * ndim
        blocksperdim[groupingDim] = dims[groupingDim]
        return self._toBlocksBySplits(blocksperdim)

    def _toBlocksBySplits(self, splitsPerDim):
        """Splits Images into ImageBlocks by subdividing the image into logically contiguous blocks.

        Parameters
        ----------
        splitsPerDim : n-tuple of positive int, with n = image dimensionality
            Each value in the splitsPerDim tuple indicates the number of subdivisions into which
            the image is to be divided along the corresponding dimension. For instance, given
            an image of dimensions x, y, z:
            * splitsPerDim (1, 1, 1) returns the original image
            * splitsPerDim (1, 1, z) divides the image into z xy planes, as would be done by
            _toBlocksByPlanes(groupingdim=2)
            * splitsPerDim (1, 2, z) divides the image into 2z blocks, with each block being half
            of an xy plane, divided in the middle of the y dimension

        """
        dims = self.dims
        ndim = len(dims)
        totnumimages = self.nimages

        if not len(splitsPerDim) == ndim:
            raise ValueError("splitsPerDim length (%d) must match image dimensionality (%d); got splitsPerDim %s" %
                             (len(splitsPerDim), ndim, str(splitsPerDim)))
        splitsPerDim = map(int, splitsPerDim)
        if any((nsplits <= 0 for nsplits in splitsPerDim)):
            raise ValueError("All numbers of blocks must be positive; got " + str(splitsPerDim))

        # slices will be sequence of sequences of slices
        # slices[i] will hold slices for ith dimension
        slices = []
        for nsplits, dimsize in zip(splitsPerDim, dims):
            blocksize = dimsize / nsplits  # integer division
            blockrem = dimsize % nsplits
            st = 0
            dimslices = []
            for blockidx in xrange(nsplits):
                en = st + blocksize
                if blockrem:
                    en += 1
                    blockrem -= 1
                dimslices.append(slice(st, min(en, dimsize), 1))
                st = en
            slices.append(dimslices)

        def _groupBySlices(imagearyval, slices_, tp_, numtp_):
            ret_vals = []
            sliceproduct = itertools.product(*slices_)
            for blockslices in sliceproduct:
                blockval = ImageBlockValue.fromArrayBySlices(imagearyval, blockslices, docopy=False)
                blockval = blockval.addDimension(newdimidx=tp_, newdimsize=numtp_)
                # resulting key will be (x, y, z) (for 3d data), where x, y, z are starting
                # position of block within image volume
                newkey = [sl.start for sl in blockslices]
                ret_vals.append((tuple(newkey), blockval))
            return ret_vals

        def _groupBySlicesAdapter(keyval):
            tpkey, imgaryval = keyval
            return _groupBySlices(imgaryval, slices, tpkey, totnumimages)

        return ImageBlocks(self.rdd.flatMap(_groupBySlicesAdapter, preservesPartitioning=False))

    def __validateOrCalcGroupingDim(self, groupingDim=None):
        """Bounds-checks the passed grouping dimension, calculating it if None is passed.

        Returns a valid grouping dimension between 0 and ndims-1, or throws ValueError if passed groupingdim is out of
        bounds.

        The calculation may trigger a spark first() call.
        """
        def calcGroupingDim(dims):
            """Returns the index of the dimension to use for grouping by image planes.

            The current heuristic is just to take the largest dimension - last largest dimension
            in case of ties.
            """
            maxd = reduce(max, dims)
            maxidxs = [i for i in xrange(len(dims)) if dims[i] == maxd]
            return maxidxs[-1]

        imgdims = self.dims
        nimgdims = len(imgdims)
        if not groupingDim is None:
            if groupingDim < -1*nimgdims or groupingDim >= nimgdims:
                raise ValueError("Grouping dimension must be between %d and %d for a %d-dimensional image; got %d" %
                                 (-1*nimgdims, nimgdims-1, nimgdims, groupingDim))
            gd = groupingDim if groupingDim >= 0 else nimgdims + groupingDim
        else:
            gd = calcGroupingDim(imgdims)
        return gd

    def __toSeriesByPlanes(self, groupingdim):
        # normalize grouping dimension, or get a reasonable grouping dimension if unspecified
        # this may trigger a first() call:
        gd = self.__validateOrCalcGroupingDim(groupingDim=groupingdim)

        # returns keys of (z, y, x); with z as grouping dimension, key values will be (0, 0, 0), (1, 0, 0), ...
        # (z-1, 0, 0)
        blocksdata = self._toBlocksByImagePlanes(groupingDim=gd)
        return blocksdata.toSeries(seriesDim=0)

    def __calcBlocksPerDim(self, blockSize):
        """Returns a partitioning strategy, represented as splits per dimension, that yields blocks
        closely matching the requested size in bytes

        Parameters
        ----------
        blockSize: positive int
            Requested size of the resulting image blocks in bytes

        Returns
        -------
        n-tuple of positive int, where n == len(self.dims)
            Each value in the returned tuple represents the number of splits to apply along the
            corresponding dimension in order to yield blocks close to the requested size.
        """
        import bisect
        minseriessize = self.nimages * dtype(self.dtype).itemsize
        dims = self.dims

        memseq = _BlockMemoryAsReversedSequence(dims)
        tmpidx = bisect.bisect_left(memseq, blockSize / float(minseriessize))
        if tmpidx == len(memseq):
            # handle case where requested block is bigger than the biggest image
            # we can produce; just give back the biggest block size
            tmpidx -= 1
        return memseq.indtosub(tmpidx)

    def _scatterToBlocks(self, blockSize="150M", blocksPerDim=None, groupingDim=None):
        if not groupingDim is None:
            # get series from blocks defined by pivoting:
            gd = self.__validateOrCalcGroupingDim(groupingDim=groupingDim)
            blocksdata = self._toBlocksByImagePlanes(groupingDim=gd)

        else:
            # get series from blocks defined by splits
            if not blocksPerDim:
                # get splits from requested block size
                if isinstance(blockSize, basestring):
                    blockSize = parseMemoryString(blockSize)
                else:
                    blockSize = int(blockSize)
                blocksPerDim = self.__calcBlocksPerDim(blockSize)
            blocksdata = self._toBlocksBySplits(blocksPerDim)

        return blocksdata

    def toSeries(self, blockSize="150M", splitsPerDim=None, groupingDim=None):
        """Converts this Images object to a Series object.

        Conversion will be performed by grouping the constituent image time points into
        smaller blocks, shuffling the blocks so that the same part of the image across time is
        processed by the same machine, and finally grouping the pixels of the image blocks
        together into a time series.

        The parameters to this method control the size of the intermediate block representation,
        which can impact performance; however results should be logically equivalent regardless of
        intermediate block size.

        Parameters
        ----------
        blockSize : positive int or string
            Requests an average size for the intermediate blocks in bytes. A passed string should
            be in a format like "256k" or "150M" (see data.parseMemoryString). If blocksPerDim
            or groupingDim are passed, they will take precedence over this argument. See
            images._BlockMemoryAsSequence for a description of the partitioning strategy used.

        splitsPerDim : n-tuple of positive int, where n = dimensionality of image
            Specifies that intermediate blocks are to be generated by splitting the i-th dimension
            of the image into splitsPerDim[i] roughly equally-sized partitions.
            1 <= splitsPerDim[i] <= self.dims[i]
            groupingDim will take precedence over this argument if both are passed.

        groupingDim : nonnegative int, 0 <= groupingDim <= len(self.dims)
            Specifies that intermediate blocks are to be generated by splitting the image
            into "planes" of dimensionality len(self.dims) - 1, along the dimension given by
            self.dims[groupingDim]. For instance, if self.dims == (x, y, z), then
            self.toSeries(groupingDim=2) would cause the images to be partioned into z intermediate
            blocks, each of size x*y and dimensionality (x, y, 1). (This is equivalent to
            self.toSeries(splitsPerDim=(1, 1, z))).

        Returns
        -------
        new Series object
        """
        blocksdata = self._scatterToBlocks(blockSize=blockSize, blocksPerDim=splitsPerDim, groupingDim=groupingDim)

        return blocksdata.toSeries(seriesDim=0)

    def saveAsBinarySeries(self, outputdirname, blockSize="150M", splitsPerDim=None, groupingDim=None,
                           overwrite=False):
        """Writes Image into files on a local filesystem, suitable for loading by SeriesLoader.fromBinary()

        The mount point specified by outputdirname must be visible to all workers; thus this method is
        primarily useful either when Spark is being run locally or in the presence of an NFS mount or
        similar shared filesystem.

        Parameters
        ----------
        outputdirname : string path or URI to directory to be created
            Output files will be written underneath outputdirname. This directory must not yet exist
            (unless overwrite is True), and must be no more than one level beneath an existing directory.
            It will be created as a result of this call.

        blockSize : positive int or string
            Requests a particular size for individual output files; see toSeries()

        splitsPerDim : n-tuple of positive int
            Specifies that output files are to be generated by splitting the i-th dimension
            of the image into splitsPerDim[i] roughly equally-sized partitions; see toSeries()

        groupingDim : nonnegative int, 0 <= groupingDim <= len(self.dims)
            Specifies that intermediate blocks are to be generated by splitting the image
            into "planes" of dimensionality len(self.dims) - 1, along the dimension given by
            self.dims[groupingDim]; see toSeries()

        overwrite : bool
            If true, outputdirname and all its contents will be deleted and recreated as part
            of this call.

        """
        writer = getParallelWriterForPath(outputdirname)(outputdirname, overwrite=overwrite)

        blocksdata = self._scatterToBlocks(blockSize=blockSize, blocksPerDim=splitsPerDim, groupingDim=groupingDim)

        binseriesrdd = blocksdata.toBinarySeries(seriesDim=0)

        def appendBin(kv):
            binlabel, binvals = kv
            return binlabel+'.bin', binvals

        binseriesrdd.map(appendBin).foreach(writer.writerFcn)

        filewriterclass = getFileWriterForPath(outputdirname)
        # write configuration file
        conf = {'input': outputdirname, 'dims': self.dims,
                'nkeys': len(self.dims), 'nvalues': self.nimages,
                'format': self.dtype, 'keyformat': 'int16'}
        confwriter = filewriterclass(outputdirname, "conf.json", overwrite=overwrite)
        confwriter.writeFile(json.dumps(conf, indent=2))

        # touch "SUCCESS" file as final action
        successwriter = filewriterclass(outputdirname, "SUCCESS", overwrite=overwrite)
        successwriter.writeFile('')

    def exportAsPngs(self, outputdirname, fileprefix="export", overwrite=False,
                     collectToDriver=True):
        """Write out basic png files for two-dimensional image data.

        Files will be written into a newly-created directory on the local file system given by outputdirname.

        All workers must be able to see the output directory via an NFS share or similar.

        Parameters
        ----------
        outputdirname : string
            Path to output directory to be created. Exception will be thrown if this directory already
            exists, unless overwrite is True. Directory must be one level below an existing directory.

        fileprefix : string
            String to prepend to all filenames. Files will be named <fileprefix>00000.png, <fileprefix>00001.png, etc

        overwrite : bool
            If true, the directory given by outputdirname will first be deleted if it already exists.

        collectToDriver : bool, default True
            If true, images will be collect()'ed at the driver first before being written out, allowing
            for use of a local filesystem at the expense of network overhead. If false, images will be written
            in parallel by each executor, presumably to a distributed or networked filesystem.
        """
        dims = self.dims
        if not len(dims) == 2:
            raise ValueError("Only two-dimensional images can be exported as .png files; image is %d-dimensional." %
                             len(dims))

        def toFilenameAndPngBuf(kv):
            key, img = kv
            fname = fileprefix+"%05d.png" % int(key)
            bytebuf = BytesIO()
            imsave(bytebuf, img, format="png")
            return fname, bytebuf.getvalue()

        bufrdd = self.rdd.map(toFilenameAndPngBuf)

        if collectToDriver:
            writer = getCollectedFileWriterForPath(outputdirname)(outputdirname, overwrite=overwrite)
            writer.writeCollectedFiles(bufrdd.collect())
        else:
            writer = getParallelWriterForPath(outputdirname)(outputdirname, overwrite=overwrite)
            bufrdd.foreach(writer.writerFcn)

    def maxProjection(self, axis=2):
        """
        Compute maximum projections of images / volumes
        along the specified dimension.

        Parameters
        ----------
        axis : int, optional, default = 2
            Which axis to compute projection along
        """
        if axis >= size(self.dims):
            raise Exception("Axis for projection (%s) exceeds image dimensions (%s-%s)" % (axis, 0, size(self.dims)-1))

        proj = self.rdd.mapValues(lambda x: amax(x, axis))
        # update dimensions to remove axis of projection
        newdims = list(self.dims)
        del newdims[axis]
        return self._constructor(proj, dims=newdims).__finalize__(self)

    def maxminProjection(self, axis=2):
        """
        Compute maximum-minimum projections of images / volumes
        along the specified dimension. This computes the sum
        of the maximum and minimum values along the given dimension.

        Parameters
        ----------
        axis : int, optional, default = 2
            Which axis to compute projection along
        """
        proj = self.rdd.mapValues(lambda x: amax(x, axis) + amin(x, axis))
        # update dimensions to remove axis of projection
        newdims = list(self.dims)
        del newdims[axis]
        return self._constructor(proj, dims=newdims).__finalize__(self)

    def subsample(self, samplefactor):
        """Downsample an image volume by an integer factor

        Parameters
        ----------
        samplefactor : positive int or tuple of positive ints
            Stride to use in subsampling. If a single int is passed, each dimension of the image
            will be downsampled by this same factor. If a tuple is passed, it must have the same
            dimensionality of the image. The strides given in a passed tuple will be applied to
            each image dimension.
        """
        dims = self.dims
        ndims = len(dims)
        if not hasattr(samplefactor, "__len__"):
            samplefactor = [samplefactor] * ndims
        samplefactor = [int(sf) for sf in samplefactor]

        if any((sf <= 0 for sf in samplefactor)):
            raise ValueError("All sampling factors must be positive; got " + str(samplefactor))

        sampleslices = [slice(0, dims[i], samplefactor[i]) for i in xrange(ndims)]
        newdims = [dims[i] / samplefactor[i] for i in xrange(ndims)]  # integer division

        def samplefunc(v, sampleslices_):
            return v[sampleslices_]

        return self._constructor(
            self.rdd.mapValues(lambda v: samplefunc(v, sampleslices)), dims=newdims).__finalize__(self)

    def planes(self, bottom, top, inclusive=True):
        """
        Subselect planes for three-dimensional image data.

        Parameters
        ----------
        bottom : int
            Bottom plane in desired selection

        top : int
            Top plane in desired selection

        inclusive : boolean, optional, default = True
            Whether returned subset of planes should include bounds
        """
        if len(self.dims) == 2 or self.dims[2] == 1:
            raise Exception("Cannot subselect planes, images must be 3D")

        if inclusive is True:
            zrange = arange(bottom, top+1)
        else:
            zrange = arange(bottom+1, top)
        newdims = [self.dims[0], self.dims[1], size(zrange)]

        return self._constructor(
            self.rdd.mapValues(lambda v: squeeze(v[:, :, zrange])), dims=newdims).__finalize__(self)

    def subtract(self, val):
        """
        Subtract a constant value or an image / volume from
        all images / volumes in the data set.

        Parameters
        ----------
        val : int, float, or ndarray
            Value to subtract
        """
        if size(val) != 1:
            if val.shape != self.dims:
                raise Exception('Cannot subtract image with dimensions %s '
                                'from images with dimension %s' % (str(val.shape), str(self.dims)))

        return self.apply(lambda x: x - val)

    def apply(self, func):
        """
        Apply a function to all images / volumes,
        otherwise perserving attributes

        Parameters
        ----------
        func : function
            Function to apply
        """
        return self._constructor(self.rdd.mapValues(func)).__finalize__(self)


class ImagesLoader(object):

    def __init__(self, sparkcontext):
        self.sc = sparkcontext

    def fromArrays(self, arrays):
        """Load Images data from passed sequence of numpy arrays.

        Expected usage is mainly in testing - having a full dataset loaded in memory
        on the driver is likely prohibitive in the use cases for which Thunder is intended.
        """
        dims = None
        dtype = None
        for ary in arrays:
            if dims is None:
                dims = ary.shape
                dtype = ary.dtype
            if not ary.shape == dims:
                raise ValueError("Arrays must all be of same shape; got both %s and %s" % (str(dims), str(ary.shape)))
            if not ary.dtype == dtype:
                raise ValueError("Arrays must all be of same data type; got both %s and %s" % (str(dtype), str(ary.dtype)))

        return Images(self.sc.parallelize(enumerate(arrays), len(arrays)), dims=dims, dtype=str(dtype), nimages=len(arrays))

    def fromStack(self, datafile, dims, ext='stack', startidx=None, stopidx=None):
        if not dims:
            raise ValueError("Image dimensions must be specified if loading from binary stack data")

        def toArray(buf):
            # previously we were casting to uint16 - still necessary?
            return frombuffer(buf, dtype='int16', count=prod(dims)).reshape(dims, order='F')

        reader = getParallelReaderForPath(datafile)(self.sc)
        readerrdd = reader.read(datafile, ext=ext, startidx=startidx, stopidx=stopidx)
        return Images(readerrdd.mapValues(toArray), nimages=reader.lastnrecs, dims=dims, dtype='int16')

    def fromTif(self, datafile, ext='tif', startidx=None, stopidx=None):
        def readTifFromBuf(buf):
            fbuf = BytesIO(buf)
            return imread(fbuf, format='tif')

        reader = getParallelReaderForPath(datafile)(self.sc)
        readerrdd = reader.read(datafile, ext=ext, startidx=startidx, stopidx=stopidx)
        return Images(readerrdd.mapValues(readTifFromBuf), nimages=reader.lastnrecs)

    def fromMultipageTif(self, datafile, ext='tif', startidx=None, stopidx=None):
        """Sets up a new Images object with data to be read from one or more multi-page tif files.

        The RDD underlying the returned Images will have key, value data as follows:

        key: int
            key is index of original data file, determined by lexicographic ordering of filenames
        value: numpy ndarray
            value dimensions with be x by y by num_channels*num_pages; all channels and pages in a file are
            concatenated together in the third dimension of the resulting ndarray. For pages 0, 1, etc
            of a multipage TIF of RGB images, ary[:,:,0] will be R channel of page 0 ("R0"), ary[:,:,1] will be B0,
            ... ary[:,:,3] == R1, and so on.

        This method attempts to explicitly import PIL. ImportError may be thrown if 'from PIL import Image' is
        unsuccessful. (PIL/pillow is not an explicit requirement for thunder.)
        """
        try:
            from PIL import Image
        except ImportError, e:
            raise ImportError("fromMultipageTif requires a successful 'from PIL import Image'; " +
                              "the PIL/pillow library appears to be missing or broken.", e)
        from matplotlib.image import pil_to_array

        def multitifReader(buf):
            fbuf = BytesIO(buf)
            multipage = Image.open(fbuf)
            pageidx = 0
            imgarys = []
            while True:
                try:
                    multipage.seek(pageidx)
                    imgarys.append(pil_to_array(multipage))
                    pageidx += 1
                except EOFError:
                    # past last page in tif
                    break
            return dstack(imgarys)

        reader = getParallelReaderForPath(datafile)(self.sc)
        readerrdd = reader.read(datafile, ext=ext, startidx=startidx, stopidx=stopidx)
        return Images(readerrdd.mapValues(multitifReader), nimages=reader.lastnrecs)

    def fromPng(self, datafile, ext='png', startidx=None, stopidx=None):
        def readPngFromBuf(buf):
            fbuf = BytesIO(buf)
            return imread(fbuf, format='png')

        reader = getParallelReaderForPath(datafile)(self.sc)
        readerrdd = reader.read(datafile, ext=ext, startidx=startidx, stopidx=stopidx)
        return Images(readerrdd.mapValues(readPngFromBuf), nimages=reader.lastnrecs)


class ImageBlocks(Data):
    """Intermediate representation used in conversion from Images to Series.

    This class is not expected to be directly used by clients.
    """

    @staticmethod
    def _blockToSeries(blockVal, seriesDim):
        for seriesKey, seriesVal in blockVal.toSeriesIter(seriesDim=seriesDim):
            yield tuple(seriesKey), seriesVal

    def toSeries(self, seriesDim=0):

        def blockToSeriesAdapter(kv):
            blockKey, blockVal = kv
            return ImageBlocks._blockToSeries(blockVal, seriesDim)

        blockedrdd = self._groupIntoSeriesBlocks()

        # returns generator of (z, y, x) array data for all z, y, x
        seriesrdd = blockedrdd.flatMap(blockToSeriesAdapter)
        return Series(seriesrdd)

    def toBinarySeries(self, seriesDim=0):

        blockedrdd = self._groupIntoSeriesBlocks()

        def blockToBinarySeries(kv):
            blockKey, blockVal = kv
            label = '-'.join(reversed(["key%02d_%05g" % (ki, k) for (ki, k) in enumerate(blockKey)]))
            keypacker = None
            buf = StringIO.StringIO()
            for seriesKey, series in ImageBlocks._blockToSeries(blockVal, seriesDim):
                if keypacker is None:
                    keypacker = struct.Struct('h'*len(seriesKey))
                # print >> sys.stderr, seriesKey, series, series.tostring().encode('hex')
                buf.write(keypacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return label, val

        return blockedrdd.map(blockToBinarySeries)

    def _groupIntoSeriesBlocks(self):
        """Combine blocks representing individual image blocks into a single time-by-blocks volume

        Returns:
        --------
        RDD, key/value: tuple of int, ImageBlockValue
        key:
            spatial indicies of start of block, for instance (x, y, z): (0, 0, 0), (0, 0, 1),... (0, 0, z_max-1)
        value:
            ImageBlockValue with single fully-populated array, dimensions of time by space, for instance (t, x, y, z):
            ary[0:t_max, :, :, z_i]
        """
        # key will be x, y, z for start of block
        # val will be single blockvalue array data with origshape and origslices in dimensions (t, x, y, z)
        # val array data will be t by (spatial block size)
        return self.rdd.groupByKey().mapValues(lambda v: ImageBlockValue.fromPlanarBlocks(v, 0))


class ImageBlockValue(object):
    """
    Helper data structure for transformations from Images to Series.

    Not intended for direct use by clients.

    Attributes
    ----------
    origshape : n-sequence of positive int
        Represents the shape of the overall volume of which this block is a component.

    origslices : n-sequence of slices
        Represents the position of this block of data within the overall volume of which the block is
        a component. Effectively, an assertion that origVolume[self.origslices] == self.values.
        Slices sl in origslices should either be equal to slice(None), representing the whole range
        of the corresponding dimension, or else should have all of sl.start, sl.stop, and sl.step
        specified.

    values : numpy ndarray
        Data making up this block.
        values.ndim will be equal to len(origshape) and len(origslices).
        values.shape will typically be smaller than origshape, but may be the same.
    """
    __slots__ = ('origshape', 'origslices', 'values')

    def __init__(self, origshape, origslices, values):
        self.origshape = origshape
        self.origslices = origslices
        self.values = values

    @classmethod
    def fromArray(cls, ary):
        return ImageBlockValue(origshape=ary.shape, origslices=tuple([slice(None)]*ary.ndim), values=ary)

    @classmethod
    def fromArrayByPlane(cls, imagearray, planedim, planeidx, docopy=False):
        """Extracts a single plane from the passed array as an ImageBlockValue.

        The origshape and origslices parameters of the returned IBV will reflect the
        original size and shape of the passed array.

        Parameters:
        -----------
        imagearray: numpy ndarray

        planedim: integer, 0 <= planedim < imagearray.ndim
            Dimension of passed array along which to extract plane

        planeidx: integer, 0 <= planeidx < imagearray.shape[planedim]
            Point along dimension from which to extract plane

        docopy: boolean, default False
            If true, return a copy of the array data; default is (typically) to return view

        Returns:
        --------
        new ImageBlockValue of plane of passed array
        """
        ndim = imagearray.ndim
        slices = [slice(None)] * ndim
        slices[planedim] = slice(planeidx, planeidx+1, 1)
        return cls.fromArrayBySlices(imagearray, slices, docopy=docopy)

    @classmethod
    def fromArrayBySlices(cls, imagearray, slices, docopy=False):
        slicedary = imagearray[slices].copy() if docopy else imagearray[slices]
        return cls(imagearray.shape, tuple(slices), slicedary)

    @classmethod
    def fromBlocks_orig(cls, blocksIter):
        """Creates a new ImageBlockValue from an iterator over blocks with compatible origshape.

        The new ImageBlockValue created will have values of shape origshape. Each block will
        copy its own data into the newly-created array according to the slicing given by its origslices
        attribute.

        """
        ary = None
        for block in blocksIter:
            if ary is None:
                ary = zeros(block.origshape, dtype=block.values.dtype)

            if not tuple(ary.shape) == tuple(block.origshape):
                raise ValueError("Shape mismatch; blocks specify differing original shapes %s and %s" %
                                 (str(block.origshape), str(ary.shape)))
            ary[block.origslices] = block.values
        return cls.fromArray(ary)

    @classmethod
    def fromPlanarBlocks(cls, blocksIter, planarDim):
        """Creates a new ImageBlockValue from an iterator over IBVs that have at least one singleton dimension.

        The resulting block will effectively be the concatenation of all blocks in the passed iterator,
        grouped as determined by each passed block's origslices attribute. This method assumes that
        the passed blocks will all have either the first or the last slice in origslices specifying only
        a single index. This index will be used to sort the consituent blocks into their proper positions
        within the returned ImageBlockValue.

        Parameters
        ----------
        blocksIter : iterable over ImageBlockValue
            All blocks in blocksIter must have identical origshape, identical values.shape, and origslices
            that are identical apart from origslices[planarDim]. For each block, origslices[planarDim]
            must specify only a single index.

        planarDim : integer
            planarDim must currently be either 0, len(origshape)-1, or -1, specifying the first, last, or
            last dimension of the block, respectively, as the dimension across which blocks are to be
            combined. origslices[planarDim] must specify a single index.

        """
        def getBlockSlices(slices, planarDim_):
            compatslices = list(slices[:])
            del compatslices[planarDim_]
            return tuple(compatslices)

        def _initialize_fromPlanarBlocks(firstBlock, planarDim_):
            # set up collection array:
            fullndim = len(firstBlock.origshape)
            if not (planarDim_ == 0 or planarDim_ == len(firstBlock.origshape)-1 or planarDim_ == -1):
                raise ValueError("planarDim must specify either first or last dimension, got %d" % planarDim_)
            if planarDim_ < 0:
                planarDim_ = fullndim + planarDim_

            newshape = list(firstBlock.values.shape[:])
            newshape[planarDim_] = block.origshape[planarDim_]

            ary = zeros(newshape, dtype=block.values.dtype)
            matchingslices = getBlockSlices(block.origslices, planarDim_)
            return ary, matchingslices, fullndim, block.origshape[:], planarDim_

        def allequal(seqA, seqB):
            return all([a == b for (a, b) in zip(seqA, seqB)])

        ary = None
        matchingslices = None
        fullndim = None
        origshape = None
        for block in blocksIter:
            if ary is None:
                # set up collection array:
                ary, matchingslices, fullndim, origshape, planarDim = \
                    _initialize_fromPlanarBlocks(block, planarDim)

            # check for compatible slices (apart from slice on planar dimension)
            blockslices = getBlockSlices(block.origslices, planarDim)
            if not allequal(matchingslices, blockslices):
                raise ValueError("Incompatible slices; got %s and %s" % (str(matchingslices), str(blockslices)))
            if not allequal(origshape, block.origshape):
                raise ValueError("Incompatible original shapes; got %s and %s" % (str(origshape), str(block.origshape)))

            # put values into collection array:
            targslices = [slice(None)] * fullndim
            targslices[planarDim] = block.origslices[planarDim]

            ary[targslices] = block.values

        # new slices should be full slice for formerly planar dimension, plus existing block slices
        newslices = list(getBlockSlices(block.origslices, planarDim))
        newslices.insert(planarDim, slice(None))

        return cls(block.origshape, newslices, ary)

    def addDimension(self, newdimidx=0, newdimsize=1):
        """Returns a new ImageBlockValue embedded in a space of dimension n+1

        Given that self is an ImageBlockValue of dimension n, returns a new
        ImageBlockValue of dimension n+1. The new IBV's origshape and origslices
        attributes will be consistent with this new dimensionality, treating the new
        IBV's values as an embedding within the new higher dimensional space.

        For instance, if ImageBlockValues are three-dimensional volumes, calling
        addDimension would recast them each as a single 'time point' in a 4-dimensional
        space.

        The new ImageBlockValue will always have the new dimension in the first position
        (index 0). ('Time' will be at self.origshape[0] and self.origslices[0]).

        Parameters:
        -----------
        newdimidx: nonnegative integer, default 0
            The zero-based index of the current block within the newly added dimension.
            (For instance, the timepoint of the volume within a new time series.)

        newdimsize: positive integer, default 1
            The total size of the newly added dimension.
            (For instance, the total number of time points in the higher-dimensional volume
            in which the new block is embedded.)

        Returns:
        --------
            a new ImageBlockValue object, with origslices and origshape modified as
            described above. New block value may be a view onto self.value.
        """
        newshape = list(self.origshape)
        newshape.insert(0, newdimsize)
        newslices = list(self.origslices)
        # todo: check array ordering here and insert at back if in 'F' order, to preserve contiguous ordering?
        newslices.insert(0, slice(newdimidx, newdimidx+1, 1))
        newblockshape = list(self.values.shape)
        newblockshape.insert(0, 1)
        newvalues = reshape(self.values, tuple(newblockshape))
        return type(self)(tuple(newshape), tuple(newslices), newvalues)

    def toSeriesIter(self, seriesDim=0):
        """Returns an iterator over key,array pairs suitable for casting into a Series object.

        Returns:
        --------
        iterator< key, series >
        key: tuple of int
        series: 1d array of self.values.dtype
        """
        rangeiters = self._get_range_iterators()
        # remove iterator over dimension where we are requesting series
        del rangeiters[seriesDim]
        # correct for original dimensionality if inserting at end of list
        insertDim = seriesDim if seriesDim >= 0 else len(self.origshape) + seriesDim
        # reverse rangeiters twice to ensure that first dimension is most rapidly varying
        for idxSeq in itertools.product(*reversed(rangeiters)):
            idxSeq = tuple(reversed(idxSeq))
            expandedIdxSeq = list(idxSeq)
            expandedIdxSeq.insert(insertDim, None)
            slices = []
            for d, (idx, origslice) in enumerate(zip(expandedIdxSeq, self.origslices)):
                if idx is None:
                    newslice = slice(None)
                else:
                    # correct slice into our own value for any offset given by origslice:
                    start = idx - origslice.start if not origslice == slice(None) else idx
                    newslice = slice(start, start+1, 1)
                slices.append(newslice)

            series = self.values[slices].squeeze()
            yield tuple(idxSeq), series

    def _get_range_iterators(self):
        """Returns a sequence of iterators over the range of the slices in self.origslices

        When passed to itertools.product, these iterators should cover the original image
        volume represented by this block.
        """
        iters = []
        noneSlice = slice(None)
        for sliceidx, sl in enumerate(self.origslices):
            if sl == noneSlice:
                it = xrange(self.origshape[sliceidx])
            else:
                it = xrange(sl.start, sl.stop, sl.step)
            iters.append(it)
        return iters

    def __repr__(self):
        return "ImageBlockValue(origshape=%s, origslices=%s, values=%s)" % \
               (repr(self.origshape), repr(self.origslices), repr(self.values))


class _BlockMemoryAsSequence(object):
    """Helper class used in calculation of slices for requested block partitions of a particular size.

    The partitioning strategy represented by objects of this class is to split into N equally-sized
    subdivisions along each dimension, starting with the rightmost dimension.

    So for instance consider an Image with spatial dimensions 5, 10, 3 in x, y, z. The first nontrivial
    partition would be to split into 2 blocks along the z axis:
    splits: (1, 1, 2)
    In this example, downstream this would turn into two blocks, one of size (5, 10, 2) and another
    of size (5, 10, 1).

    The next partition would be to split into 3 blocks along the z axis, which happens to
    corresponding to having a single block per z-plane:
    splits: (1, 1, 3)
    Here these splits would yield 3 blocks, each of size (5, 10, 1).

    After this the z-axis cannot be partitioned further, so the next partition starts splitting along
    the y-axis:
    splits: (1, 2, 3)
    This yields 6 blocks, each of size (5, 5, 1).

    Several other splits are possible along the y-axis, going from (1, 2, 3) up to (1, 10, 3).
    Following this we move on to the x-axis, starting with splits (2, 10, 3) and going up to
    (5, 10, 3), which is the finest subdivision possible for this data.

    Instances of this class represent the average size of a block yielded by this partitioning
    strategy in a linear order, moving from the most coarse subdivision (1, 1, 1) to the finest
    (x, y, z), where (x, y, z) are the dimensions of the array being partitioned.

    This representation is intended to support binary search for the partitioning strategy yielding
    a block size closest to a requested amount.
    """
    def __init__(self, dims):
        self._dims = dims

    def indtosub(self, idx):
        """Converts a linear index to a corresponding partition strategy, represented as
        number of splits along each dimension.
        """
        dims = self._dims
        ndims = len(dims)
        sub = [1] * ndims
        for didx, d in enumerate(dims[::-1]):
            didx = ndims - (didx + 1)
            delta = min(dims[didx]-1, idx)
            if delta > 0:
                sub[didx] += delta
                idx -= delta
            if idx <= 0:
                break
        return tuple(sub)

    def blockMemoryForSplits(self, sub):
        """Returns the average number of cells in a block generated by the passed sequence of splits.
        """
        from operator import mul
        sz = [d / float(s) for (d, s) in zip(self._dims, sub)]
        return reduce(mul, sz)

    def __len__(self):
        return sum([d-1 for d in self._dims]) + 1

    def __getitem__(self, item):
        sub = self.indtosub(item)
        return self.blockMemoryForSplits(sub)


class _BlockMemoryAsReversedSequence(_BlockMemoryAsSequence):
    """A version of _BlockMemoryAsSequence that represents the linear ordering of splits in the
    opposite order, starting with the finest partitioning allowable for the array dimensions.

    This can yield a sequence of block sizes in increasing order, which is required for binary
    search using python's 'bisect' library.
    """
    def _reverseIdx(self, idx):
        l = len(self)
        if idx < 0 or idx >= l:
            raise IndexError("list index out of range")
        return l - (idx + 1)

    def indtosub(self, idx):
        return super(_BlockMemoryAsReversedSequence, self).indtosub(self._reverseIdx(idx))

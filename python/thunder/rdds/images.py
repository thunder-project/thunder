from numpy import ndarray, arange, amax, amin, size, squeeze

from thunder.rdds.data import Data
from thunder.rdds.keys import Dimensions


class Images(Data):
    """
    Distributed collection of images or volumes.

    Backed by an RDD of key-value pairs, where the key
    is an identifier and the value is a two or three-dimensional array.
    """

    _metadata = Data._metadata + ['_dims', '_nimages']

    def __init__(self, rdd, dims=None, nimages=None, dtype=None):
        super(Images, self).__init__(rdd, dtype=dtype)
        if dims and not isinstance(dims, Dimensions):
            try:
                dims = Dimensions.fromTuple(dims)
            except:
                raise TypeError("Images dims parameter must be castable to Dimensions object, got: %s" % str(dims))
        self._dims = dims
        self._nimages = nimages

    @property
    def dims(self):
        if self._dims is None:
            self.populateParamsFromFirstRecord()
        return self._dims

    @property
    def nimages(self):
        if self._nimages is None:
            self._nimages = self.rdd.count()
        return self._nimages

    @property
    def dtype(self):
        # override just calls superclass; here for explicitness
        return super(Images, self).dtype

    @property
    def _constructor(self):
        return Images

    def populateParamsFromFirstRecord(self):
        record = super(Images, self).populateParamsFromFirstRecord()
        self._dims = Dimensions.fromTuple(record[1].shape)
        return record

    def _resetCounts(self):
        self._nimages = None
        return self

    @staticmethod
    def _check_type(record):
        if not isinstance(record[0], tuple):
            raise Exception('Keys must be tuples')
        if not isinstance(record[1], ndarray):
            raise Exception('Values must be ndarrays')

    def partition(self, partitioningStrategy):
        partitioningStrategy.setImages(self)
        returntype = partitioningStrategy.getPartitionedImagesClass()
        vals = self.rdd.flatMap(partitioningStrategy.partitionFunction, preservesPartitioning=False)
        # fastest changing dimension (e.g. x) is first, so must sort reversed keys to get desired ordering
        # sort must come after group, b/c group will mess with ordering.
        #groupedvals = vals.groupByKey(numPartitions=partitioningStrategy.npartitions).sortBy(lambda (k, _): k[::-1])
        groupedvals = vals.groupBy(lambda (k, _): k.spatialKey).sortBy(lambda (k, _): tuple(k[::-1]))
        # groupedvals is now rdd of (z, y, x spatial key, [(partitioning key, numpy array)...]
        blockedvals = groupedvals.map(partitioningStrategy.blockingFunction)
        return returntype(blockedvals, dims=self.dims, nimages=self.nimages, dtype=self.dtype)

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

        from matplotlib.pyplot import imsave
        from io import BytesIO
        from thunder.rdds.fileio.writers import getParallelWriterForPath, getCollectedFileWriterForPath

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

        return self._constructor(
            self.rdd.mapValues(lambda v: v[sampleslices]), dims=newdims).__finalize__(self)

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

        return self._constructor(self.rdd.mapValues(lambda v: squeeze(v[:, :, zrange])),
                                 dims=newdims).__finalize__(self)

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


class PartitioningStrategy(object):
    """Superclass for objects that define ways to split up images into smaller blocks.
    """
    def __init__(self):
        self._dims = None
        self._nimages = None
        self._dtype = None

    @property
    def dims(self):
        """Shape of the Images data to which this PartitioningStrategy is to be applied.

        dims will be taken from the Images passed in the last call to setImages().

        n-tuple of positive int, or None if setImages has not been called
        """
        return self._dims

    @property
    def nimages(self):
        """Number of images (time points) in the Images data to which this PartitioningStrategy is to be applied.

        nimages will be taken from the Images passed in the last call to setImages().

        positive int, or None if setImages has not been called
        """
        return self._nimages

    @property
    def dtype(self):
        """Numpy data type of the Images data to which this PartitioningStrategy is to be applied.

        String or numpy dtype, or None if setImages has not been called
        """
        return self.dtype

    def setImages(self, images):
        """Readies the PartitioningStrategy to operate over the passed Images object.

        dims, nimages, and dtype will be initialized by this call.

        No return value.
        """
        self._dims = images.dims
        self._nimages = images.nimages
        self._dtype = images.dtype

    def getPartitionedImagesClass(self):
        """Get the subtype of PartitionedImages that instances of this strategy will produce.

        Subclasses should override this method to return the appropriate PartitionedImages subclass.
        """
        return PartitionedImages

    def partitionFunction(self, timePointIdxAndImageArray):
        raise NotImplementedError("partitionFunction not implemented")

    def blockingFunction(self, spatialIdxAndPartitionedSequence):
        raise NotImplementedError("blockingFunction not implemented")

    @property
    def npartitions(self):
        """The number of Spark partitions across which the resulting RDD is to be distributed.
        """
        raise NotImplementedError("numPartitions not implemented")


class PartitioningKey(object):
    @property
    def temporalKey(self):
        raise NotImplementedError

    @property
    def spatialKey(self):
        raise NotImplementedError


class PartitionedImages(Data):
    """Superclass for data returned by an Images.partition() call.
    """
    _metadata = Data._metadata + ['_dims', '_nimages']

    def __init__(self, rdd, dims=None, nimages=None, dtype=None):
        super(PartitionedImages, self).__init__(rdd, dtype=dtype)
        self._dims = dims
        self._nimages = nimages

    @property
    def dims(self):
        """Shape of the original Images data from which this PartitionedImages was derived.

        n-tuple of positive int
        """
        if not self._dims:
            self.populateParamsFromFirstRecord()
        return self._dims

    @property
    def nimages(self):
        """Number of images (time points) in the original Images data from which this PartitionedImages was derived.

        positive int
        """
        return self._nimages

    def toSeries(self):
        """Returns a Series Data object.

        Subclasses that can be converted to a Series object are expected to override this method.
        """
        raise NotImplementedError("toSeries not implemented")

    def toBinarySeries(self):
        """Returns an RDD of binary series data.

        The keys of a binary series RDD should be filenames ending in ".bin".
        The values should be packed binary data.

        Subclasses that can be converted to a Series object are expected to override this method.
        """
        raise NotImplementedError

    def saveAsBinarySeries(self, outputdirname, overwrite=False):
        """Writes out Series-formatted data.

        Subclasses are *not* expected to override this method.

        Parameters
        ----------
        outputdirname : string path or URI to directory to be created
            Output files will be written underneath outputdirname. This directory must not yet exist
            (unless overwrite is True), and must be no more than one level beneath an existing directory.
            It will be created as a result of this call.

        overwrite : bool
            If true, outputdirname and all its contents will be deleted and recreated as part
            of this call.
        """
        from thunder.rdds.fileio.writers import getParallelWriterForPath
        from thunder.rdds.fileio.seriesloader import writeSeriesConfig

        writer = getParallelWriterForPath(outputdirname)(outputdirname, overwrite=overwrite)

        binseriesrdd = self.toBinarySeries()

        binseriesrdd.foreach(writer.writerFcn)
        writeSeriesConfig(outputdirname, len(self.dims), self.nimages, dims=self.dims.count,
                          keytype='int16', valuetype=self.dtype, overwrite=overwrite)

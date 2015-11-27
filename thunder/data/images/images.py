from numpy import ndarray, arange, amax, amin, size

from ..base import Data
from ..keys import Dimensions


class Images(Data):
    """
    Distributed collection of images or volumes.

    Backed by an RDD of key-value pairs, where the key
    is an identifier and the value is a two or three-dimensional array.
    """
    _metadata = Data._metadata + ['_dims']

    def __init__(self, rdd, dims=None, nrecords=None, dtype=None):
        super(Images, self).__init__(rdd, nrecords=nrecords, dtype=dtype)
        if dims and not isinstance(dims, Dimensions):
            try:
                dims = Dimensions.fromTuple(dims)
            except:
                raise TypeError("Images dims parameter must be castable to Dimensions object, got: %s" % str(dims))
        self._dims = dims

    @property
    def dims(self):
        if self._dims is None:
            self.fromfirst()
        return self._dims

    @property
    def shape(self):
        if self._shape is None:
            self._shape = (self.nrecords,) + self.dims.count
        return self._shape

    @property
    def dtype(self):
        # override just calls superclass; here for explicitness
        return super(Images, self).dtype

    @property
    def _constructor(self):
        return Images

    def fromfirst(self):
        record = super(Images, self).fromfirst()
        self._dims = Dimensions.fromTuple(record[1].shape)
        return record

    @staticmethod
    def _check_type(record):
        if not isinstance(record[0], tuple):
            raise Exception('Keys must be tuples')
        if not isinstance(record[1], ndarray):
            raise Exception('Values must be ndarrays')

    def toBlocks(self, size="150M", units="pixels", padding=0):
        """
        Convert to Blocks, each representing a subdivision of the larger Images data.

        Parameters
        ----------
        size : string memory size, tuple of splits per dimension, or instance of BlockingStrategy
            String interpreted as memory size (e.g. "64M"). Tuple of ints interpreted as
            "pixels per dimension" (default) or "splits per dimension", depending on units.
            Instance of BlockingStrategy can be passed directly.

        units : string, either "pixels" or "splits", default = "pixels"
            What units to use for a tuple size.

        padding : non-negative integer or tuple of int, optional, default = 0
            Will generate blocks with extra `padding` voxels along each dimension.
            Padded voxels will overlap with those in neighboring blocks, but will not be included
            when converting blocks to Series or Images.

        Returns
        -------
        Blocks instance
        """
        from thunder.data.blocks.strategy import BlockingStrategy, SimpleBlockingStrategy, PaddedBlockingStrategy

        stratClass = SimpleBlockingStrategy if not padding else PaddedBlockingStrategy

        if isinstance(size, BlockingStrategy):
            blockingStrategy = size
        elif isinstance(size, basestring) or isinstance(size, int):
            # make blocks close to the desired size
            blockingStrategy = stratClass.generateFromBlockSize(self, size, padding=padding)
        else:
            # assume it is a tuple of positive int specifying splits
            blockingStrategy = stratClass(size, units=units, padding=padding)

        blockingStrategy.setSource(self)
        avgSize = blockingStrategy.calcAverageBlockSize()
        if avgSize >= BlockingStrategy.DEFAULT_MAX_BLOCK_SIZE:
            # TODO: use logging module here rather than print
            print "Thunder WARNING: average block size of %g bytes exceeds suggested max size of %g bytes" % \
                  (avgSize, BlockingStrategy.DEFAULT_MAX_BLOCK_SIZE)

        returntype = blockingStrategy.getBlocksClass()
        vals = self.rdd.flatMap(blockingStrategy.blockingFunction, preservesPartitioning=False)
        # fastest changing dimension (e.g. x) is first, so must sort reversed keys to get desired ordering
        # sort must come after group, b/c group will mess with ordering.
        groupedvals = vals.groupBy(lambda (k, _): k.spatialKey).sortBy(lambda (k, _): tuple(k[::-1]))
        # groupedvals is now rdd of (z, y, x spatial key, [(partitioning key, numpy array)...]
        blockedvals = groupedvals.map(blockingStrategy.combiningFunction)
        return returntype(blockedvals, dims=self.dims, nimages=self.nrecords, dtype=self.dtype)

    def toTimeSeries(self, size="150M"):
        """
        Converts this Images object to a TimeSeries object.

        This method is equivalent to images.asBlocks(size).asSeries().asTimeSeries().

        Parameters
        ----------
        size: string memory size, optional, default = "150M"
            String interpreted as memory size (e.g. "64M").

        units: string, either "pixels" or "splits", default = "pixels"
            What units to use for a tuple size.

        Returns
        -------
        new TimeSeries object

        See also
        --------
        Images.toBlocks
        """
        return self.toBlocks(size).toSeries().totimeseries()

    def toSeries(self, size="150M"):
        """
        Converts this Images object to a Series object.

        This method is equivalent to images.toBlocks(size).toSeries().

        Parameters
        ----------
        size: string memory size, optional, default = "150M"
            String interpreted as memory size (e.g. "64M").

        Returns
        -------
        new Series object

        See also
        --------
        Images.toBlocks
        """
        return self.toBlocks(size).toSeries()

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
        newDims = list(self.dims)
        del newDims[axis]
        return self._constructor(proj, dims=newDims).__finalize__(self)

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
        newDims = list(self.dims)
        del newDims[axis]
        return self._constructor(proj, dims=newDims).__finalize__(self)

    def subsample(self, sampleFactor):
        """
        Downsample an image volume by an integer factor

        Parameters
        ----------
        sampleFactor : positive int or tuple of positive ints
            Stride to use in subsampling. If a single int is passed, each dimension of the image
            will be downsampled by this same factor. If a tuple is passed, it must have the same
            dimensionality of the image. The strides given in a passed tuple will be applied to
            each image dimension.
        """
        dims = self.dims
        ndims = len(dims)
        if not hasattr(sampleFactor, "__len__"):
            sampleFactor = [sampleFactor] * ndims
        sampleFactor = [int(sf) for sf in sampleFactor]

        if any((sf <= 0 for sf in sampleFactor)):
            raise ValueError("All sampling factors must be positive; got " + str(sampleFactor))

        def divRoundup(a, b):
            # thanks stack overflow & Eli Collins:
            # http://stackoverflow.com/questions/7181757/how-to-implement-division-with-round-towards-infinity-in-python
            # this only works for positive ints, but we've checked for that above
            return (a + b - 1) // b

        sampleSlices = [slice(0, dims[i], sampleFactor[i]) for i in xrange(ndims)]
        newDims = [divRoundup(dims[i], sampleFactor[i]) for i in xrange(ndims)]

        return self._constructor(
            self.rdd.mapValues(lambda v: v[sampleSlices]), dims=newDims).__finalize__(self)
            
    def gaussianFilter(self, sigma=2, order=0):
        """
        Spatially smooth images with a gaussian filter.

        Filtering will be applied to every image in the collection and can be applied
        to either images or volumes. For volumes, if an single scalar sigma is passed,
        it will be interpreted as the filter size in x and y, with no filtering in z.

        parameters
        ----------
        sigma : scalar or sequence of scalars, default=2
            Size of the filter size as standard deviation in pixels. A sequence is interpreted
            as the standard deviation for each axis. For three-dimensional data, a single
            scalar is interpreted as the standard deviation in x and y, with no filtering in z.

        order : choice of 0 / 1 / 2 / 3 or sequence from same set, optional, default = 0
            Order of the gaussian kernel, 0 is a gaussian, higher numbers correspond
            to derivatives of a gaussian.
        """
        from scipy.ndimage.filters import gaussian_filter

        dims = self.dims
        ndims = len(dims)

        if ndims == 3 and size(sigma) == 1:
            sigma = [sigma, sigma, 0]

        return self._constructor(
            self.rdd.mapValues(lambda v: gaussian_filter(v, sigma, order))).__finalize__(self)

    def uniformFilter(self, size=2):
        """
        Spatially filter images using a uniform filter.

        Filtering will be applied to every image in the collection and can be applied
        to either images or volumes. For volumes, if an single scalar neighborhood is passed,
        it will be interpreted as the filter size in x and y, with no filtering in z.

        parameters
        ----------
        size: int, optional, default=2
            Size of the filter neighbourhood in pixels. A sequence is interpreted
            as the neighborhood size for each axis. For three-dimensional data, a single
            scalar is intrepreted as the neighborhood in x and y, with no filtering in z.
        """
        return self._imageFilter(filter='uniform', size=size)

    def medianFilter(self, size=2):
        """
        Spatially filter images using a median filter.

        Filtering will be applied to every image in the collection and can be applied
        to either images or volumes. For volumes, if an single scalar neighborhood is passed,
        it will be interpreted as the filter size in x and y, with no filtering in z.

        parameters
        ----------
        size: int, optional, default=2
            Size of the filter neighbourhood in pixels. A sequence is interpreted
            as the neighborhood size for each axis. For three-dimensional data, a single
            scalar is intrepreted as the neighborhood in x and y, with no filtering in z.
        """
        return self._imageFilter(filter='median', size=size)

    def _imageFilter(self, filter=None, size=2):
        """
        Generic function for applying a filtering operation to images or volumes.

        See also
        --------
        Images.uniformFilter
        Images.medianFilter
        """
        from numpy import isscalar
        from scipy.ndimage.filters import median_filter, uniform_filter

        FILTERS = {
            'median': median_filter,
            'uniform': uniform_filter
        }

        func = FILTERS[filter]

        dims = self.dims
        ndims = len(dims)

        if ndims == 3 and isscalar(size) == 1:
            def filter_(im):
                im.setflags(write=True)
                for z in arange(0, dims[2]):
                    im[:, :, z] = func(im[:, :, z], size)
                return im
        else:
            filter_ = lambda x: func(x, size)

        return self._constructor(
            self.rdd.mapValues(lambda v: filter_(v))).__finalize__(self)

    def localcorr(self, neighborhood=2):
        """
        Correlate every pixel to the average of its local neighborhood.

        This algorithm computes, for every spatial record, the correlation coefficient
        between that record's series, and the average series of all records within
        a local neighborhood with a size defined by the neighborhood parameter.
        The neighborhood is currently required to be a single integer, which represents the neighborhood
        size in both x and y.

        parameters
        ----------
        neighborhood: int, optional, default=2
            Size of the correlation neighborhood (in both the x and y directions), in pixels.
        """

        if not isinstance(neighborhood, int):
            raise ValueError("The neighborhood must be specified as an integer.")

        from numpy import corrcoef

        nimages = self.nrecords

        # Spatially average the original image set over the specified neighborhood
        blurred = self.uniformFilter((neighborhood * 2) + 1)

        # Union the averaged images with the originals to create an Images object containing 2N images (where
        # N is the original number of images), ordered such that the first N images are the averaged ones.
        combined = self.rdd.union(blurred.apply_keys(lambda k: k + nimages).rdd)
        combinedImages = self._constructor(combined, nrecords=(2 * nimages)).__finalize__(self)

        # Correlate the first N (averaged) records with the last N (original) records
        series = combinedImages.toSeries()
        corr = series.apply_values(lambda x: corrcoef(x[:nimages], x[nimages:])[0, 1])

        return corr.pack()

    def crop(self, minbound, maxbound):
        """
        Crop a spatial region from 2D or 3D data.

        Parameters
        ----------
        minbound : list or tuple
            Minimum of crop region (x,y) or (x,y,z)

        maxbound : list or tuple
            Maximum of crop region (x,y) or (x,y,z)

        Returns
        -------
        Images object with cropped images / volume
        """
        dims = self.dims
        ndims = len(dims)
        dimsCount = dims.count

        if ndims < 2 or ndims > 3:
            raise Exception("Cropping only supported on 2D or 3D image data.")

        dimMinMaxTuples = zip(dimsCount, minbound, maxbound)
        if len(dimMinMaxTuples) != ndims:
            raise ValueError("Number of specified bounds (%d) must equal image dimensionality (%d)" % 
                             (len(dimMinMaxTuples), ndims))
        slices = []
        newdims = []
        for dim, minb, maxb in dimMinMaxTuples:
            if maxb > dim:
                raise ValueError("Maximum bound (%d) may not exceed image size (%d)" % (maxb, dim))
            if minb < 0:
                raise ValueError("Minumum bound (%d) must be positive" % minb)
            if minb < maxb:
                slise = slice(minb, maxb)
                newdims.append(maxb - minb)
            elif minb == maxb:
                slise = minb  # just an integer index, not a slice; this squeezes out singleton dimensions
                # don't append to newdims, this dimension will be squeezed out
            else:
                raise ValueError("Minimum bound (%d) must be <= max bound (%d)" % (minb, maxb))
            slices.append(slise)

        newrdd = self.rdd.mapValues(lambda v: v[slices])
        newdims = tuple(newdims)

        return self._constructor(newrdd, dims=newdims).__finalize__(self)

    def planes(self, start, stop):
        """
        Subselect planes from 3D image data.

        Parameters
        ----------
        start, stop : int
            Indices of region to crop in z, interpreted according to python slice indexing conventions.

        See also
        --------
        Images.crop
        """

        dims = self.dims

        if len(dims) == 2 or dims[2] == 1:
            raise Exception("Cannot subselect planes, images must be 3D")

        return self.crop([0, 0, start], [dims[0], dims[1], stop])

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
            if val.shape != self.dims.count:
                raise Exception('Cannot subtract image with dimensions %s '
                                'from images with dimension %s' % (str(val.shape), str(self.dims)))

        return self.apply_values(lambda x: x - val)

    def renumber(self):
        """
        Recalculates keys for this Images object.

        New keys will be a sequence of consecutive integers, starting at 0 and ending at self.nrecords-1.
        """
        renumberedRdd = self.rdd.values().zipWithIndex().map(lambda (ary, idx): (idx, ary))
        return self._constructor(renumberedRdd).__finalize__(self)

    def topng(self, path, prefix="image", overwrite=False):
        """
        Write 2d or 3d images as PNG files.

        Files will be written into a newly-created directory.
        Three-dimensional data will be treated as RGB channels.

        Parameters
        ----------
        path : string
            Path to output directory, must be one level below an existing directory.

        prefix : string
            String to prepend to filenames.

        overwrite : bool
            If true, the directory given by path will first be deleted if it exists.
        """
        from thunder.data.images.writers import topng
        # TODO add back colormap and vmin/vmax
        topng(self, path, prefix=prefix, overwrite=overwrite)

    def totif(self, path, prefix="image", overwrite=False):
        """
        Write 2d or 3d images as TIF files.

        Files will be written into a newly-created directory.
        Three-dimensional data will be treated as RGB channels.

        Parameters
        ----------
        path : string
            Path to output directory, must be one level below an existing directory.

        prefix : string
            String to prepend to filenames.

        overwrite : bool
            If true, the directory given by path will first be deleted if it exists.
        """
        from thunder.data.images.writers import totif
        # TODO add back colormap and vmin/vmax
        totif(self, path, prefix=prefix, overwrite=overwrite)

    def tobinary(self, path, prefix="image", overwrite=False):
        """
        Write out images or volumes as flat binary files.

        Files will be written into a newly-created directory.

        Parameters
        ----------
        path : string
            Path to output directory, must be one level below an existing directory.

        prefix : string
            String to prepend to filenames.

        overwrite : bool
            If true, the directory given by path will first be deleted if it exists.
        """
        from thunder.data.images.writers import tobinary
        tobinary(self, path, prefix=prefix, overwrite=overwrite)
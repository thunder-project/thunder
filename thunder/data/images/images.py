from numpy import ndarray, arange, amax, amin, size, asarray

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
            self.populateParamsFromFirstRecord()
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

    def populateParamsFromFirstRecord(self):
        record = super(Images, self).populateParamsFromFirstRecord()
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
        return self.toBlocks(size).toSeries().toTimeSeries()

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

    def localCorr(self, neighborhood=2):
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
        combined = self.rdd.union(blurred.applyKeys(lambda k: k + nimages).rdd)
        combinedImages = self._constructor(combined, nrecords=(2 * nimages)).__finalize__(self)

        # Correlate the first N (averaged) records with the last N (original) records
        series = combinedImages.toSeries()
        corr = series.applyValues(lambda x: corrcoef(x[:nimages], x[nimages:])[0, 1])

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

    def meanByRegions(self, selection):
        """
        Reduces images to one or more spatially averaged values using the given selection, which can be
        either a mask array or sequence of indicies.

        A passed mask must be a numpy ndarray of the same shape as the individual arrays in this
        Images object. If the mask array is of integer or unsigned integer type, one mean value will
        be calculated for each unique nonzero value in the passed mask. (That is, all pixels with a
        value of '1' in the mask will be averaged together, as will all with a mask value of '2', and so
        on.) For other mask array types, all nonzero values in the mask will be averaged together into
        a single regional average.

        Alternatively, subscripted indices may be passed directly as a sequence of sequences of tuple indicies. For
        instance, selection=[[(0,1), (1,0)], [(2,1), (2,2)]] would return two means, one for the region made up
        of the pixels at (0,1) and (1,0), and the other of (2,1) and (2,2).

        The returned object will be a new 2d Images object with dimensions (1, number of regions). This can be
        converted into a Series object and from there into time series arrays by calling
        regionMeanImages.toSeries().collect().

        Parameters
        ----------
        selection: ndarray mask with shape equal to self.dims.count, or sequence of sequences of pixel indicies

        Returns
        -------
        new Images object
        """
        from numpy import array, mean
        ctx = self.rdd.context

        def meanByIntMask(kv):
            key, ary = kv
            uniq = bcUnique.value
            msk = bcSelection.value
            meanVals = [mean(ary[msk == grp]) for grp in uniq if grp != 0]
            return key, array(meanVals, dtype=ary.dtype).reshape((1, -1))

        def meanByMaskIndices(kv):
            key, ary = kv
            maskIdxsSeq = bcSelection.value
            means = array([mean(ary[maskIdxs]) for maskIdxs in maskIdxsSeq], dtype=ary.dtype).reshape((1, -1))
            return key, means

        # argument type checking
        if isinstance(selection, ndarray):
            # passed a numpy array mask
            from numpy import unique
            # getting image dimensions just requires a first() call, not too expensive; and we probably
            # already have them anyway
            if selection.shape != self.dims.count:
                raise ValueError("Shape mismatch between mask '%s' and image dimensions '%s'; shapes must be equal" %
                                 (str(selection.shape), str(self.dims.count)))

            if selection.dtype.kind in ('i', 'u'):
                # integer or unsigned int mask
                selectFcn = meanByIntMask
                uniq = unique(selection)
                nregions = len(uniq) - 1 if 0 in uniq else len(uniq)  # 0 doesn't turn into a region
                bcUnique = ctx.broadcast(uniq)
                bcSelection = ctx.broadcast(selection)
            else:
                selectFcn = meanByMaskIndices
                nregions = 1
                bcUnique = None
                bcSelection = ctx.broadcast((selection.nonzero(), ))
        else:
            # expect sequence of sequences of subindices if we aren't passed a mask
            selectFcn = meanByMaskIndices
            regionSelections = []
            imgNDims = len(self.dims.count)
            for regionIdxs in selection:
                # generate sequence of subindex arrays
                # instead of sequence [(x0, y0, z0), (x1, y1, z1), ... (xN, yN, zN)], want:
                # array([x0, x1, ... xN]), array([y0, y1, ... yN]), ... array([z0, z1, ... zN])
                # this can be used directly in an array indexing expression: ary[regionSelect]
                for idxTuple in regionIdxs:
                    if len(idxTuple) != imgNDims:
                        raise ValueError("Image is %d-dimensional, but got %d dimensional index: %s" %
                                         (imgNDims, len(idxTuple), str(idxTuple)))
                regionSelect = []
                regionIdxs = asarray(regionIdxs).T.tolist()
                for idxDimNum, dimIdxs in enumerate(zip(regionIdxs)):
                    imgDimMax = self.dims.count[idxDimNum]
                    dimIdxAry = array(dimIdxs, dtype='uint16')
                    idxMin, idxMax = dimIdxAry.min(), dimIdxAry.max()
                    if idxMin < 0 or idxMax >= imgDimMax:
                        raise ValueError("Index of dimension %d out of bounds; " % idxDimNum +
                                         "got min/max %d/%d, all must be >=0 and <%d" %
                                         (idxMin, idxMax, imgDimMax))
                    regionSelect.append(dimIdxAry)
                regionSelections.append(regionSelect)
            nregions = len(regionSelections)
            bcUnique = None
            bcSelection = ctx.broadcast(regionSelections)

        data = self.rdd.map(selectFcn)
        return self._constructor(data, dims=(1, nregions)).__finalize__(self)

    def planes(self, startidz, stopidz):
        """
        Subselect planes from 3D image data.

        Parameters
        ----------
        startidz, stopidz : int
            Indices of region to crop in z, interpreted according to python slice indexing conventions.

        See also
        --------
        Images.crop
        """

        dims = self.dims

        if len(dims) == 2 or dims[2] == 1:
            raise Exception("Cannot subselect planes, images must be 3D")

        return self.crop([0, 0, startidz], [dims[0], dims[1], stopidz])

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

        return self.applyValues(lambda x: x - val)

    def renumber(self):
        """
        Recalculates keys for this Images object.

        New keys will be a sequence of consecutive integers, starting at 0 and ending at self.nrecords-1.
        """
        renumberedRdd = self.rdd.values().zipWithIndex().map(lambda (ary, idx): (idx, ary))
        return self._constructor(renumberedRdd).__finalize__(self)

    def toPng(self, outputDirPath, cmap=None, vmin=None, vmax=None, prefix="image", overwrite=False):
        """
        Write images to PNG files
        """
        from thunder.data.images.writers import toPng
        toPng(self, outputDirPath, cmap, vmin=vmin, vmax=vmax, prefix=prefix, overwrite=overwrite)

    def toBinary(self, outputDirPath, prefix="image", overwrite=False):
        """
        Write images to binary files
        """
        from thunder.data.images.writers import toBinary
        toBinary(self, outputDirPath, prefix=prefix, overwrite=overwrite)
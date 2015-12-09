from numpy import ndarray, arange, amax, amin, size, asarray, random

from ..base import Data


class Images(Data):
    """
    Collection of images or volumes.
    """
    _metadata = Data._metadata

    def __init__(self, values, mode='local'):
        super(Images, self).__init__(values, mode=mode)

    @property
    def _constructor(self):
        return Images

    @property
    def dims(self):
        return self.shape[1:]

    def toblocks(self, size="150M", units="pixels", padding=0):
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
        from thunder.data.blocks.strategy import \
            BlockingStrategy, SimpleBlockingStrategy, PaddedBlockingStrategy

        stratClass = SimpleBlockingStrategy if not padding else PaddedBlockingStrategy

        if isinstance(size, BlockingStrategy):
            blockingStrategy = size
        elif isinstance(size, basestring) or isinstance(size, int):
            # make blocks close to the desired size
            blockingStrategy = stratClass.fromsize(self, size, padding=padding)
        else:
            # assume it is a tuple of positive int specifying splits
            blockingStrategy = stratClass(size, units=units, padding=padding)

        blockingStrategy.setsource(self)
        returntype = blockingStrategy.getclass()
        vals = self.rdd.flatMap(blockingStrategy.blocker, preservesPartitioning=False)
        # fastest changing dimension (e.g. x) is first, so must sort reversed keys
        # sort must come after group, b/c group will mess with ordering.
        groupedvals = vals.groupBy(lambda (k, _): k.spatial).sortBy(lambda (k, _): tuple(k[::-1]))
        # groupedvals is now rdd of (z, y, x spatial key, [(partitioning key, numpy array)...]
        blockedvals = groupedvals.map(blockingStrategy.combiner)
        return returntype(blockedvals, dims=self.dims, nimages=self.nrecords, dtype=self.dtype)

    def totimeseries(self, size="150M"):
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
        return self.toseries().totimeseries()

    def toseries(self, size="150"):
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
        from thunder.data.series.series import Series

        n = len(self.shape) - 1

        if self.mode == 'spark':
            return Series(self.values.swap((0,), tuple(range(n)), size=size))

        if self.mode == 'local':
            return Series(self.values.transpose(tuple(range(1, n+1)) + (0,)))

    def tolocal(self):
        """
        Convert to local representation.
        """
        from thunder.data.images.readers import fromarray

        if self.mode == 'local':
            raise ValueError('images already in local mode')

        return fromarray(self.toarray())

    def tospark(self, engine=None):
        """
        Convert to spark representation.
        """
        from thunder.data.images.readers import fromarray

        if self.mode == 'spark':
            raise ValueError('images already in spark mode')

        return fromarray(self.toarray(), engine=engine)

    def toarray(self):
        """
        Return a local array
        """
        out = asarray(self.values)
        if out.shape[0] == 1:
            out = out.squeeze(axis=0)
        return out

    def foreach(self, func):
        """
        Execute a function on each image
        """
        if self.mode == 'spark':
            self.values.tordd().map(lambda (k, v): (k[0], v)).foreach(func)
        else:
            [func(kv) for kv in enumerate(self.values)]

    def sample(self, nsamples=100, seed=None):
        """
        Extract random sample of series.

        Parameters
        ----------
        nsamples : int, optional, default = 100
            The number of data points to sample.

        seed : int, optional, default = None
            Random seed.
        """
        if nsamples < 1:
            raise ValueError("number of samples must be larger than 0, got '%g'" % nsamples)

        if seed is None:
            seed = random.randint(0, 2 ** 32)

        if self.mode == 'spark':
            result = asarray(self.values.tordd().values().takeSample(False, nsamples, seed))

        else:
            inds = [int(k) for k in random.rand(nsamples) * self.shape[0]]
            result = asarray([self.values[i] for i in inds])

        return self._constructor(result)

    def map(self, func, dims=None):
        """
        Map a function to each image
        """
        if dims is None:
            dims = self.dims
        return self._map(func, axis=0, value_shape=dims)

    def filter(self, func):
        """
        Filter images
        """
        return self._filter(func, axis=0)

    def reduce(self, func):
        """
        Reduce over images
        """
        return self._reduce(func, axis=0)

    def mean(self):
        """
        Compute the mean across images
        """
        return self._constructor(self.values.mean(axis=0, keepdims=True))

    def var(self):
        """
        Compute the variance across images
        """
        return self._constructor(self.values.var(axis=0, keepdims=True))

    def std(self):
        """
        Compute the standard deviation across images
        """
        return self._constructor(self.values.std(axis=0, keepdims=True))

    def sum(self):
        """
        Compute the sum across images
        """
        return self._constructor(self.values.sum(axis=0, keepdims=True))

    def max(self):
        """
        Compute the max across images
        """
        return self._constructor(self.values.max(axis=0, keepdims=True))

    def min(self):
        """
        Compute the min across images
        """
        return self._constructor(self.values.min(axis=0, keepdims=True))

    def max_projection(self, axis=2):
        """
        Compute maximum projections of images / volumes
        along the specified dimension.

        Parameters
        ----------
        axis : int, optional, default = 2
            Which axis to compute projection along
        """
        if axis >= size(self.dims):
            raise Exception("Axis for projection (%s) exceeds "
                            "image dimensions (%s-%s)" % (axis, 0, size(self.dims)-1))

        newdims = list(self.dims)
        del newdims[axis]
        return self.map(lambda x: amax(x, axis), dims=newdims)

    def max_min_projection(self, axis=2):
        """
        Compute maximum-minimum projections of images / volumes
        along the specified dimension. This computes the sum
        of the maximum and minimum values along the given dimension.

        Parameters
        ----------
        axis : int, optional, default = 2
            Which axis to compute projection along
        """
        if axis >= size(self.dims):
            raise Exception("Axis for projection (%s) exceeds "
                            "image dimensions (%s-%s)" % (axis, 0, size(self.dims)-1))

        newdims = list(self.dims)
        del newdims[axis]
        return self.map(lambda x: amax(x, axis) + amin(x, axis), dims=newdims)

    def subsample(self, factor):
        """
        Downsample an image volume by an integer factor

        Parameters
        ----------
        sample_factor : positive int or tuple of positive ints
            Stride to use in subsampling. If a single int is passed, each dimension of the image
            will be downsampled by this same factor. If a tuple is passed, it must have the same
            dimensionality of the image. The strides given in a passed tuple will be applied to
            each image dimension.
        """
        dims = self.dims
        ndims = len(dims)
        if not hasattr(factor, "__len__"):
            factor = [factor] * ndims
        factor = [int(sf) for sf in factor]

        if any((sf <= 0 for sf in factor)):
            raise ValueError("All sampling factors must be positive; got " + str(factor))

        def roundup(a, b):
            return (a + b - 1) // b

        slices = [slice(0, dims[i], factor[i]) for i in xrange(ndims)]
        newdims = tuple([roundup(dims[i], factor[i]) for i in xrange(ndims)])

        return self.map(lambda v: v[slices], dims=newdims)

    def gaussian_filter(self, sigma=2, order=0):
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

        return self.map(lambda v: gaussian_filter(v, sigma, order), dims=self.dims)

    def uniform_filter(self, size=2):
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
        return self._image_filter(filter='uniform', size=size)

    def median_filter(self, size=2):
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
        return self._image_filter(filter='median', size=size)

    def _image_filter(self, filter=None, size=2):
        """
        Generic function for maping a filtering operation to images or volumes.

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

        return self.map(lambda v: filter_(v), dims=self.dims)

    def localcorr(self, neighborhood=2):
        """
        Correlate every pixel to the average of its local neighborhood.

        This algorithm computes, for every spatial record, the correlation coefficient
        between that record's series, and the average series of all records within
        a local neighborhood with a size defined by the neighborhood parameter.
        The neighborhood is currently required to be a single integer,
        which represents the neighborhood size in both x and y.

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

        # Union the averaged images with the originals to create an
        # Images object containing 2N images (where N is the original number of images),
        # ordered such that the first N images are the averaged ones.
        combined = self.rdd.union(blurred.map_keys(lambda k: k + nimages).rdd)
        combinedImages = self._constructor(combined, nrecords=(2 * nimages)).__finalize__(self)

        # Correlate the first N (averaged) records with the last N (original) records
        series = combinedImages.toseries()
        corr = series.map_values(lambda x: corrcoef(x[:nimages], x[nimages:])[0, 1])

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

        if ndims < 2 or ndims > 3:
            raise Exception("Cropping only supported on 2D or 3D image data.")

        pairs = zip(dims, minbound, maxbound)
        if len(pairs) != ndims:
            raise ValueError("Number of bounds (%d) must equal image dimensionality (%d)" %
                             (len(pairs), ndims))
        slices = []
        newdims = []
        for dim, minb, maxb in pairs:
            if maxb > dim:
                raise ValueError("Maximum bound (%d) may not exceed image size (%d)" % (maxb, dim))
            if minb < 0:
                raise ValueError("Minumum bound (%d) must be positive" % minb)
            if minb < maxb:
                s = slice(minb, maxb)
                newdims.append(maxb - minb)
            elif minb == maxb:
                s = minb
            else:
                raise ValueError("Minimum bound (%d) must be <= max bound (%d)" % (minb, maxb))
            slices.append(s)

        newdims = tuple(newdims)

        return self.map(lambda v: v[slices], dims=newdims)

    def planes(self, start, stop):
        """
        Subselect planes from 3D image data.

        Parameters
        ----------
        start, stop : int
            Indices of region to crop in z, according to python slice indexing conventions.

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
        if isinstance(val, ndarray):
            if val.shape != self.dims:
                raise Exception('Cannot subtract image with dimensions %s '
                                'from images with dimension %s' % (str(val.shape), str(self.dims)))

        return self.map(lambda x: x - val, dims=self.dims)

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
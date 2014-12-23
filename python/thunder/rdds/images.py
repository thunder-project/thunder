from numpy import ndarray, arange, amax, amin, size, squeeze, dtype

from thunder.rdds.data import Data
from thunder.rdds.keys import Dimensions
from thunder.utils.common import parseMemoryString


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
        # splitsPerDim is expected to be in the dimensions ordering convention
        import itertools
        from thunder.rdds.imageblocks import ImageBlocks, ImageBlockValue

        dims = self.dims.count[:]  # currently in Dimensions-convention
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

        # reverse slices to be in numpy shape ordering convention:
        # slices = slices[::-1]

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

        return ImageBlocks(self.rdd.flatMap(_groupBySlicesAdapter, preservesPartitioning=False), dtype=self.dtype)

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
                blockSize = parseMemoryString(blockSize)
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
            be in a format like "256k" or "150M" (see util.common.parseMemoryString). If blocksPerDim
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
        from thunder.rdds.fileio.writers import getParallelWriterForPath
        from thunder.rdds.fileio.seriesloader import writeSeriesConfig

        writer = getParallelWriterForPath(outputdirname)(outputdirname, overwrite=overwrite)

        blocksdata = self._scatterToBlocks(blockSize=blockSize, blocksPerDim=splitsPerDim, groupingDim=groupingDim)

        binseriesrdd = blocksdata.toBinarySeries(seriesDim=0)

        def appendBin(kv):
            binlabel, binvals = kv
            return binlabel+'.bin', binvals

        binseriesrdd.map(appendBin).foreach(writer.writerFcn)
        writeSeriesConfig(outputdirname, len(self.dims), self.nimages, dims=self.dims.count,
                          keytype='int16', valuetype=self.dtype, overwrite=overwrite)

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

        def div_roundup(a, b):
            # thanks stack overflow & Eli Collins:
            # http://stackoverflow.com/questions/7181757/how-to-implement-division-with-round-towards-infinity-in-python
            # this only works for positive ints, but we've checked for that above
            return (a + b - 1) // b

        sampleslices = [slice(0, dims[i], samplefactor[i]) for i in xrange(ndims)]
        newdims = [div_roundup(dims[i] ,samplefactor[i]) for i in xrange(ndims)]

        return self._constructor(
            self.rdd.mapValues(lambda v: v[sampleslices]), dims=newdims).__finalize__(self)
            
    def gaussianFilter(self, sigma=2):
        """Spatially smooth images using a gaussian filter.

        This function will be applied to every image in the data set and can be applied
        to either images or volumes. In the latter case, filtering will be applied separately
        to each plane.

        parameters
        ----------
        sigma : int, optional, default=2
            Size of the filter neighbourhood in pixels
        """

        from scipy.ndimage.filters import gaussian_filter

        dims = self.dims
        ndims = len(dims)

        if ndims == 2:

            def filter(im):
                return gaussian_filter(im, sigma)

        if ndims == 3:

            def filter(im):
                im.setflags(write=True)
                for z in arange(0, dims[2]):
                    im[:, :, z] = gaussian_filter(im[:, :, z], sigma)
                return im

        return self._constructor(
            self.rdd.mapValues(lambda v: filter(v))).__finalize__(self)

    def medianFilter(self, size=2):
        """Spatially smooth images using a median filter.

        The filtering will be applied to every image in the collection and can be applied
        to either images or volumes. In the latter case, filtering will be applied separately
        to each plane.

        parameters
        ----------
        size: int, optional, default=2
            Size of the filter neighbourhood in pixels
        """

        from scipy.ndimage.filters import median_filter

        dims = self.dims
        ndims = len(dims)

        if ndims == 2:

            def filter(im):
                return median_filter(im, size)

        if ndims == 3:

            def filter(im):
                im.setflags(write=True)
                for z in arange(0, dims[2]):
                    im[:, :, z] = median_filter(im[:, :, z], size)
                return im

        return self._constructor(
            self.rdd.mapValues(lambda v: filter(v))).__finalize__(self)

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

        if len(zrange) == 0:
            raise Exception("No planes selected with range (%g, %g) and inclusive=%s, "
                            "try a different range" % (bottom, top, inclusive))

        if zrange.min() < self.dims.min[2]:
            raise Exception("Cannot include plane %g, first plane is %g" % (zrange.min(), self.dims.min[2]))

        if zrange.max() > self.dims.max[2]:
            raise Exception("Cannout include plane %g, last plane is %g" % (zrange.max(), self.dims.max[2]))

        newdims = [self.dims[0], self.dims[1], size(zrange)]

        if size(zrange) < 2:
            newdims = newdims[0:2]

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
            if val.shape != self.dims.count:
                raise Exception('Cannot subtract image with dimensions %s '
                                'from images with dimension %s' % (str(val.shape), str(self.dims)))

        return self.applyValues(lambda x: x - val)


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

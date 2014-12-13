"""Classes relating to Blocks, which represent subdivided Images data.
"""
import cStringIO as StringIO
import itertools
from numpy import zeros, arange
import struct
from thunder.rdds.data import Data
from thunder.rdds.keys import Dimensions


def getStartStopStep(slise, refsize):
    """Extracts the start, stop, and step parameters from a passed slice.

    If the slice is slice(None), returns refsize as the stop position. start will be 0 and
    step 1 in this case, which corresponds to normal python slice semantics.
    """
    stop = refsize if slise.stop is None else slise.stop
    start = 0 if slise.start is None else slise.start
    step = 1 if slise.step is None else slise.step
    return start, stop, step


def sliceToXRange(sl, stop):
    return xrange(*getStartStopStep(sl, stop))


def slicesToIterators(slices, shape):
    """Returns a sequence of iterators over the range of the passed slices.

    The output of this function is expected to be passed into itertools.product.

    Parameters
    ----------
    slices: sequence of slices

    shape: tuple of positive int, same length as slices

    Returns
    -------
    sequence of iterators, one iterator per slice
    """
    return [sliceToXRange(slise, size) for slise, size in zip(slices, shape)]


class Blocks(Data):
    """Superclass for subdivisions of Images data.

    Subclasses of Blocks will be returned by an images.toBlocks() call.
    """
    _metadata = Data._metadata + ['_dims', '_nimages']

    def __init__(self, rdd, dims=None, nimages=None, dtype=None):
        super(Blocks, self).__init__(rdd, dtype=dtype)
        self._dims = dims
        self._nimages = nimages

    @property
    def dims(self):
        """Shape of the original Images data from which these Blocks were derived.

        n-tuple of positive int
        """
        if not self._dims:
            self.populateParamsFromFirstRecord()
        return self._dims

    @property
    def nimages(self):
        """Number of images (time points) in the original Images data from which these Blocks were derived.

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


class SimpleBlocks(Blocks):
    """Basic concrete implementation of Blocks.

    These Blocks will be contiguous, nonoverlapping subsections of the original Images arrays.
    """
    _metadata = Data._metadata + ['_dims', '_nimages']

    @property
    def _constructor(self):
        return SimpleBlocks

    def populateParamsFromFirstRecord(self):
        record = super(SimpleBlocks, self).populateParamsFromFirstRecord()
        self._dims = Dimensions.fromTuple(record[0].origshape)
        return record

    @staticmethod
    def toTimeSlicedBlocks(kv):
        """Generator function that yields an iteration over (timepoint, ImageBlockValue)
        pairs.
        """
        blockkey, ary = kv

        planarrange = sliceToXRange(blockkey.imgslices[0], blockkey.origshape[0])
        for tpidx in planarrange:
            # set up new slices:
            newimgslices = blockkey.imgslices[1:]
            neworigshape = blockkey.origshape[1:]
            # new array value:
            newval = ary[tpidx]
            newkey = ImageReconstructionKey(tpidx, neworigshape, newimgslices)
            yield newkey, newval

    @staticmethod
    def combineTimeSlicedBlocks(temporalIdxAndSlicedSequence):
        temporalIdx, slicedSequence = temporalIdxAndSlicedSequence
        # sequence will be of (partitioning key, np array) pairs
        ary = None
        for key, block in slicedSequence:
            if ary is None:
                # set up collection array:
                ary = zeros(key.origshape, block.dtype)
            # put values into collection array:
            ary[key.imgslices] = block

        return temporalIdx, ary

    def toSeries(self, newdtype="smallfloat", casting="safe"):
        from thunder.rdds.series import Series
        # returns generator of (z, y, x) array data for all z, y, x
        seriesrdd = self.rdd.flatMap(lambda kv: kv[0].toSeriesIter(kv[1]))

        idx = arange(self._nimages) if self._nimages else None
        return Series(seriesrdd, dims=self.dims, index=idx, dtype=self.dtype).astype(newdtype, casting=casting)

    def toImages(self):
        from thunder.rdds.images import Images
        timerdd = self.rdd.flatMap(SimpleBlocks.toTimeSlicedBlocks)
        timesortedrdd = timerdd.groupBy(lambda (k, _): k.temporalKey).sortByKey()
        imagesrdd = timesortedrdd.map(SimpleBlocks.combineTimeSlicedBlocks)
        return Images(imagesrdd, dims=self._dims, nimages=self._nimages, dtype=self._dtype)

    @staticmethod
    def getBinarySeriesNameForKey(blockKey):
        """

        Returns
        -------
        string blocklabel
            Block label will be in form "key02_0000k-key01_0000j-key00_0000i" for i,j,k x,y,z indicies as first series
            in block. No extension (e.g. ".bin") is appended to the block label.
        """
        return '-'.join(reversed(["key%02d_%05g" % (ki, k) for (ki, k) in enumerate(blockKey)]))

    def toBinarySeries(self):

        def blockToBinarySeries(kv):
            blockKey, blockVal = kv
            label = SimpleBlocks.getBinarySeriesNameForKey(blockKey.spatialKey)+".bin"
            keypacker = None
            buf = StringIO.StringIO()
            for seriesKey, series in blockKey.toSeriesIter(blockVal):
                if keypacker is None:
                    keypacker = struct.Struct('h'*len(seriesKey))
                # print >> sys.stderr, seriesKey, series, series.tostring().encode('hex')
                buf.write(keypacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return label, val

        return self.rdd.map(blockToBinarySeries)


class PaddedBlocks(SimpleBlocks):

    @property
    def _constructor(self):
        return PaddedBlocks

    @staticmethod
    def toSeriesIter(blockingKey, ary):
        """Returns an iterator over key,array pairs suitable for casting into a Series object.

        Returns:
        --------
        iterator< key, series >
        key: tuple of int
        series: 1d array of self.values.dtype
        """
        rangeiters = SimpleBlocks._get_range_iterators(blockingKey.imgslices, blockingKey.origshape)
        # remove iterator over temporal dimension where we are requesting series
        del rangeiters[0]
        insertDim = 0

        for idxSeq in itertools.product(*reversed(rangeiters)):
            expandedIdxSeq = list(reversed(idxSeq))
            expandedIdxSeq.insert(insertDim, None)
            slices = []
            for d, (idx, origslice, valslice) in enumerate(zip(expandedIdxSeq, blockingKey.imgslices, blockingKey.valslices)):
                if idx is None:
                    newslice = slice(None)
                else:
                    # correct slice into our own value for any offset given by origslice:
                    # start = idx - origslice.start if not origslice == slice(None) else idx
                    # newslice = slice(start, start+1, 1)\
                    valstart, valstop, valstep = \
                        PaddedBlocks.getStartStopStep(valslice, blockingKey.valshape[d])
                    imgstart, imgstop, imgstep = \
                        PaddedBlocks.getStartStopStep(origslice, blockingKey.imgshape[d])
                    stepnum = (idx - imgstart) / imgstep
                    validx = valstart + stepnum * valstep
                    newslice = slice(validx, validx+1, 1)
                slices.append(newslice)

            series = ary[slices].squeeze()
            yield tuple(reversed(idxSeq)), series


class BlockingKey(object):
    @property
    def temporalKey(self):
        raise NotImplementedError

    @property
    def spatialKey(self):
        raise NotImplementedError


class ImageReconstructionKey(BlockingKey):
    def __init__(self, timeidx, origshape, imgslices):
        self.timeidx = timeidx
        self.origshape = origshape
        self.imgslices = imgslices

    @property
    def temporalKey(self):
        return self.timeidx

    @property
    def spatialKey(self):
        # should this be reversed?
        return tuple(self.imgslices)

    def __repr__(self):
        return "ImageReconstructionKey(timeidx=%d, origshape=%s, imgslices=%s)" % (self.timeidx,
                                                                                   self.origshape,
                                                                                   self.imgslices)


class BlockGroupingKey(BlockingKey):
    def __init__(self, origshape, imgslices):
        self.origshape = origshape
        self.imgslices = imgslices

    @property
    def temporalKey(self):
        # temporal key is index of time point, obtained from first slice (time dimension)
        return self.imgslices[0].start

    @property
    def spatialKey(self):
        # should this be reversed?
        return tuple(sl.start for sl in self.imgslices[1:])

    def getSeriesDataForSpatialIndices(self, spatialIndices, ary):
        """Returns a one-dimensional array corresponding to the time series (dimension 0) at the passed spatial
        indices, given in original image coordinates

        Parameters
        ----------
        spatialIndices: sequence of int of length == ary.ndim - 1
            spatial coordinates in original image space, ordered as x, y, z
        """
        slices = [slice(None)]  # start with full slice in temporal dimension, dim 0
        for idx, origslice in zip(spatialIndices, self.imgslices[1:]):
            # transform index from original image space into block array space
            # requires any offset in image space to be subtracted from our starting position
            start = idx - origslice.start if not (origslice.start is None) else idx
            newslice = slice(start, start+1, 1)
            slices.append(newslice)
        return ary[slices].squeeze()

    def toSeriesIter(self, ary):
        """Returns an iterator over key, array pairs suitable for casting into a Series object.

        Returns:
        --------
        iterator< key, series >
        key: tuple of int
        series: 1d array of self.values.dtype
        """
        rangeiters = slicesToIterators(self.imgslices[1:], self.origshape[1:])
        for idxSeq in itertools.product(*reversed(rangeiters)):
            spatialIndices = tuple(reversed(idxSeq))
            series = self.getSeriesDataForSpatialIndices(spatialIndices, ary)
            yield spatialIndices, series

    def __repr__(self):
        return "BlockGroupingKey(origshape=%s, imgslices=%s)" % (self.origshape, self.imgslices)


class PaddedBlockGroupingKey(BlockGroupingKey):
    """Key used to extract and sort PaddedBlocks.

    These keys are expected to be used in a Spark key-value RDD, where the value is a numpy array representing
    a contiguous subsection of a 4d spatiotemporal image.

    Attributes
    ----------
    origshape: sequence of positive int
        Shape of original Images array of which this block is a part. This shape includes a "time" dimension as the
        first dimension. (This additional dimension is not present on Images values, which are each assumed to represent
        a single point in time.

    imgslices: sequence of slices
        Slices into an array of shape origshape; these slices represent the full extent of the block in its original
        space, including padding.

    coreimgslices: sequence of slices
        Slices into an array of shape origshape; these slices represent the 'core' block, without padding, in its
        original space.

    corevalslices: sequence of slices
        Slices into the array-type value for which this object is the key in a Spark key-value RDD. These slices
        represent the 'core' block, without padding. These slices should be the same size as those in coreimgslices,
        differing only in their offsets.
    """
    def __init__(self, origshape, padimgslices, imgslices, valshape, valslices):
        super(PaddedBlockGroupingKey, self).__init__(origshape, imgslices)
        self.padimgslices = padimgslices
        self.valshape = valshape
        self.valslices = valslices

    def __repr__(self):
        return "PaddedBlockGroupingKey(origshape=%s, padimgslices=%s, imgslices=%s, valshape=%s, valslices=%s)" % \
               (self.origshape, self.padimgslices, self.imgslices, self.valshape, self.valslices)
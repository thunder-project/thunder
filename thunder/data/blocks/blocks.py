from numpy import zeros, arange, multiply, tile, add
import cStringIO as StringIO
import itertools
import struct

from ..base import Data
from ..keys import Dimensions


def slice_to_tuple(s, size):
    """
    Extracts the start, stop, and step parameters from a passed slice.

    If the slice is slice(None), returns size as the stop position. Start will be 0 and
    step 1 in this case, which corresponds to normal python slice semantics.
    """
    stop = size if s.stop is None else s.stop
    start = 0 if s.start is None else s.start
    step = 1 if s.step is None else s.step
    return start, stop, step

def slice_to_range(s, stop):
    return xrange(*slice_to_tuple(s, stop))

def slices_to_iterators(slices, shape):
    """
    Returns a sequence of iterators over the range of the passed slices.

    The output of this function is expected to be passed into itertools.product.

    Parameters
    ----------
    slices : tuple
        Sequence of slices

    shape : tuple of int
        Same length as slices

    Returns
    -------
    sequence of iterators, one iterator per slice
    """
    return [slice_to_range(s, size) for s, size in zip(slices, shape)]

def record_iter(key, value):
    """
    Generator function that iterates over records..

    Parameters:
    -----------
    key : BlockGroupingKey
        Key associated with the passed array value

    value : ndarray
        Array data for the block associated with the passed key

    Yields:
    --------
    iterator < key, series >
    key : tuple of int
    series : 1d array of self.values.dtype
    """
    for spatialIndices in key.spatial_range():
        series = key.getseries(spatialIndices, value)
        yield spatialIndices, series

def block_iter(key, value):
    """
    Generator function that iterates over block arrays.

    Parameters:
    -----------
    key : BlockGroupingKey
        Key associated with the passed array value

    value : ndarrary
        Array data for the block associated with the passed key

    Yields:
    -------
    iterator <ImageReconstructionKey, ndarray>

    ImageReconstructionKey : new key
        Key imgSlices, origShape will be for full Image space, not just block,
        and will not include time

    array : ndarray with dimensions x, y, z equal to block shape; no time dimension
    """
    slices = key.slices[1:]
    shape = key.shape[1:]
    for tpIdx in key.temporal_range():
        newvalue = key.getimage(tpIdx, value)
        newkey = ImageReconstructionKey(tpIdx, shape, slices)
        yield newkey, newvalue


class Blocks(Data):
    """
    Superclass for subdivisions of Images data.

    Subclasses of Blocks will be returned by an images.toBlocks() call.
    """
    _metadata = Data._metadata + ['_dims', '_nimages']

    def __init__(self, rdd, dims=None, nimages=None, dtype=None):
        super(Blocks, self).__init__(rdd, dtype=dtype)
        self._dims = dims
        self._nimages = nimages

    @property
    def dims(self):
        """
        Shape of the original Images data from which these Blocks were derived.
        """
        if not self._dims:
            self.fromfirst()
        return self._dims

    @property
    def shape(self):
        """
        Total shape
        """
        if self._shape is None:
            self._shape = (self.nrecords,) + self.dims.count
        return self._shape

    @property
    def nimages(self):
        """
        Number of images in the Images data from which these Blocks were derived.

        positive int
        """
        return self._nimages

    def _reset(self):
        pass

    def toseries(self):
        """
        Returns a Series Data object.

        Subclasses that can be converted to a Series object are expected to override this method.
        """
        raise NotImplementedError("toSeries not implemented")

    def getbinary(self):
        """
        Extract chunks of binary data for each block.

        The keys of each chunk should be filenames ending in ".bin".
        The values should be packed binary data.

        Subclasses that can be converted to a Series object are expected to override this method.
        """
        raise NotImplementedError("getbinary not implemented")

    def tobinary(self, path, overwrite=False):
        """
        Writes out Series-formatted binary data.

        Subclasses are *not* expected to override this method.

        Parameters
        ----------
        path : string
            Output files will be written underneath path.
            This directory must not yet exist (unless overwrite is True),
            and must be no more than one level beneath an existing directory.
            It will be created as a result of this call.

        overwrite : bool
            If true, outputdirname and all its contents will
            be deleted and recreated as part of this call.
        """
        from thunder import credentials
        from thunder.data.writers import get_parallel_writer
        from thunder.data.series.writers import write_config

        if not overwrite:
            from thunder.utils.common import check_path
            check_path(path, credentials=credentials())
            overwrite = True

        writer = get_parallel_writer(path)(path, overwrite=overwrite, credentials=credentials())
        binary = self.getbinary()
        binary.foreach(writer.write)
        write_config(path, len(self.dims), self.nimages,
                     keytype='int16', valuetype=self.dtype, overwrite=overwrite)


class SimpleBlocks(Blocks):
    """
    Basic concrete implementation of Blocks.

    These Blocks will be contiguous, nonoverlapping subsections of the original Images arrays.
    """
    @property
    def _constructor(self):
        return SimpleBlocks

    def fromfirst(self):
        record = super(SimpleBlocks, self).fromfirst()
        self._dims = Dimensions.fromTuple(record[0].shape)
        return record

    def toseries(self, newtype='smallfloat', casting='safe'):
        from thunder.data.series.series import Series
        data = self.rdd.flatMap(lambda kv: record_iter(kv[0], kv[1]))
        index = arange(self._nimages) if self._nimages else None
        series = Series(data, dims=self.dims, index=index, dtype=self.dtype)
        return series.astype(newtype, casting=casting)

    def toimages(self):
        """
        Convert blocks to images
        """
        from thunder.data.images.images import Images

        def combine(kv):
            index, sequence = kv
            ary = None
            for key, block in sequence:
                if ary is None:
                    ary = zeros(key.shape, block.dtype)
                ary[key.slices] = block

            return index, ary

        data = self.rdd.flatMap(lambda kv: block_iter(kv[0], kv[1]))
        datasorted = data.groupBy(lambda (k, _): k.temporal).sortByKey()
        out = datasorted.map(combine)
        return Images(out, dims=self._dims, nrecords=self._nimages, dtype=self._dtype)

    def getbinary(self):
        """
        Extract binary data from each block
        """
        from thunder.data.series.writers import getlabel

        def getblock(kv):
            key, val = kv
            label = getlabel(key.spatial)+".bin"
            packer = None
            buf = StringIO.StringIO()
            for seriesKey, series in record_iter(key, val):
                if packer is None:
                    packer = struct.Struct('h'*len(seriesKey))
                buf.write(packer.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return label, val

        return self.rdd.map(getblock)


class PaddedBlocks(SimpleBlocks):

    @property
    def _constructor(self):
        return PaddedBlocks


class BlockingKey(object):
    @property
    def temporal(self):
        raise NotImplementedError

    @property
    def spatial(self):
        raise NotImplementedError


class ImageReconstructionKey(BlockingKey):
    def __init__(self, time, shape, slices):
        self.time = time
        self.shape = shape
        self.slices = slices

    @property
    def temporal(self):
        return self.time

    @property
    def spatial(self):
        return tuple(self.slices)

    def __repr__(self):
        return "ImageReconstructionKey\ntime: %d\nshape: %s\nslices: %s" % \
               (self.time, self.shape, self.slices)


class BlockGroupingKey(BlockingKey):
    """
    Key used to extract and sort SimpleBlocks.

    These keys are expected to be used in a collection where the value
    is an ndarray representing a contiguous subsection of a 4d spatiotemporal image.

    Attributes
    ----------
    shape : sequence of positive int
        Shape of original Images array of which this block is a part.
        This shape includes a "time" dimension as the first dimension.
        (This additional dimension is not present on Images values,
        which are each assumed to represent a single point in time.)

    slices : sequence of slices
        Slices into an array of shape origShape; these slices represent
        the full extent of the block in its original space.
        These slices include the temporal dimension as the first
        slice in the sequence (imgSlices[0]).

    pixels_per_dim : list or tuple
        Pixels per dimension, from the blocking strategy used to generate this block
    """
    def __init__(self, shape, slices, pixels_per_dim=None):
        self.shape = shape
        self.slices = slices
        self.pixels_per_dim = tuple(pixels_per_dim) if pixels_per_dim is not None else None

    @property
    def temporal(self):
        start = self.slices[0].start
        return start if not (start is None) else 0

    @property
    def spatial(self):
        return tuple(sl.start for sl in self.slices[1:])

    @property
    def block_shape(self):
        return tuple(sl.stop - sl.start for sl in self.slices[1:])

    def neighbors(self):
        """
        Construct list of spatial keys that neighbor this one.

        Uses the current block's spatial key, and the pixels
        per block dimension (derived from the blocking strategy)
        to construct a list of neighbors. Excludes any neighbors
        that would fall outside the bounds, assumed to range from
        (0, 0, ...) to the original shape.
        """
        center = self.spatial
        extent = self.pixels_per_dim
        maxbound = self.shape[1:]

        ndim = len(maxbound)
        minbound = zeros(ndim, 'int').tolist()
        origin = tuple(zeros(ndim, 'int'))
        shifts = tuple(tile([-1, 0, 1], (ndim, 1)).tolist())

        neighbors = []
        import itertools
        for shift in itertools.product(*shifts):
            if not (shift == origin):
                newkey = add(center, multiply(extent, shift))
                cond1 = sum(map(lambda (x, y): x < y, zip(newkey, minbound)))
                cond2 = sum(map(lambda (x, y): x >= y, zip(newkey, maxbound)))
                if not (cond1 or cond2):
                    neighbors.append(tuple(newkey))

        return neighbors

    def concatenated(self):
        """
        Returns a new key that is a copy of self,
        except that the new temporal range of imgSlices will be
        from 0 to the total number of time points.

        Used in SimpleBlockingStrategy.combiningFcn to generate new key
        for a concatenated set of spatially-aligned
        blocks grouped together across time.
        """
        # new slices should be full slice for planes, plus existing block slices
        newslices = [slice(0, self.shape[0])] + list(self.slices)[1:]
        return BlockGroupingKey(shape=self.shape, slices=tuple(newslices),
                                pixels_per_dim=self.pixels_per_dim)

    def getseries(self, indices, ary):
        """
        Returns a one-dimensional array corresponding to the time series
        (dimension 0) at the passed spatial indices, given in original image coordinates

        Parameters
        ----------
        indices: sequence of int
            Spatial coordinates in original image space, ordered as x, y, z.
            Length should be ary.ndim - 1
        """
        slices = [slice(None)]
        for index, original in zip(indices, self.slices[1:]):
            # transform index from original image space into block array space
            # requires any offset in image space to be subtracted from our starting position
            start = index - original.start if not (original.start is None) else index
            newslice = slice(start, start+1, 1)
            slices.append(newslice)
        return ary[slices].squeeze()

    def getimage(self, index, value):
        return value[index]

    def spatial_range(self):
        """
        Generator yielding spatial indices as (x, y, z) index tuples.

        The indices yielded by this method will cover the full spatial
        extent of the block with which this key is associated.

        The indices will be ordered so that the last index (z-dimension)
        will be incrementing most quickly. This should
        make these indices suitable to be cast directly
        into keys for Series data objects.

        The indices yielded here can be passed into self.getseries()
        to get the associated 1d time series array.

        Yields
        ------
        spatial indices, tuple of nonnegative int
        """
        iters = slices_to_iterators(self.slices[1:], self.shape[1:])
        for indices in itertools.product(*reversed(iters)):
            yield tuple(reversed(indices))

    def temporal_range(self):
        """
        Returns an iterable object over the range of time points represented by this key.

        Returns
        -------
        iterable over nonnegative int
        """
        return slice_to_range(self.slices[0], self.shape[0])

    def __repr__(self):
        return "BlockGroupingKey\nshape: %s\nslices: %s" % (self.shape, self.slices)


class PaddedBlockGroupingKey(BlockGroupingKey):
    """
    Key used to extract and sort PaddedBlocks.

    These keys are expected to be used in a collection where the value
    is an ndarray representing a contiguous subsection of a 4d spatiotemporal image.

    Attributes
    ----------
    shape : sequence of positive int
        Shape of original Images array of which this block is a part.
        This shape includes a "time" dimension as the first dimension.
        (This additional dimension is not present on Images values,
        which are each assumed to represent a single point in time.)

    slices : sequence of slices
        Slices into an array with dimensions shape; these slices represent
        the 'core' block, without padding, in its original space.
        These slices include the temporal dimension as the first slice in the sequence.

    padded_slices : sequence of slices
        Slices into an array with dimensions shape; these slices represent the full
        extent of the block in its original space, including padding.
        These slices include the temporal dimension as the first slice in the sequence

    array_shape : tuple of positive int
        Shape of associated numpy array value

    array_slices : sequence of slices
        Slices into the array-type value for which this object
        is the key in a collection. These slices represent the 'core' block, without padding.
        These slices should be the same size as those in slices, differing only in their offsets.

    pixels_per_dim : tuple
        Pixels per dimension, from the blocking strategy used to generate this block
    """
    def __init__(self, shape, slices, padded_slices, array_shape, array_slices, pixels_per_dim):
        super(PaddedBlockGroupingKey, self).__init__(shape, slices, pixels_per_dim)
        self.padded_slices = padded_slices
        self.array_shape = array_shape
        self.array_slices = array_slices

    @property
    def padding(self):
        before = tuple([self.slices[i].start - self.padded_slices[i].start
                        for i in range(1, len(self.shape))])
        after = tuple([self.padded_slices[i].stop - self.slices[i].stop
                       for i in range(1, len(self.shape))])
        return before, after

    def concatenated(self):
        """
        Returns a new key that is a copy of self, except that the
        new temporal range is from 0 to the total number of time points.
        """
        allslices = slice(0, self.shape[0])
        slices = tuple([allslices] + list(self.slices[1:]))
        padded_slices = tuple([allslices] + list(self.padded_slices[1:]))
        array_shape = tuple([self.shape[0]] + list(self.array_shape[1:]))
        array_slices = tuple([allslices] + list(self.array_slices[1:]))
        return PaddedBlockGroupingKey(self.shape, slices, padded_slices,
                                      array_shape, array_slices, self.pixels_per_dim)

    def getseries(self, indices, value):
        """
        Returns a one-dimensional array corresponding to the time series
        (dimension 0) at the passed spatial indices, given in original image coordinates

        This implementation overrides that in BlockGroupingKey,
        taking into account padding on the block side as described in self.array_slices.

        Parameters
        ----------
        indices : sequence of int
            Spatial coordinates in original image space, ordered as x, y, z.
            Should have length ary.ndim - 1.
        """
        slices = [slice(None)]
        zipped = zip(indices, self.slices[1:], self.shape[1:],
                     self.array_slices[1:], self.array_shape[1:])
        for index, original, shape, array_slice, array_shape in zipped:
            start, stop, step = slice_to_tuple(original, shape)
            array_start, array_stop, array_step = slice_to_tuple(array_slice, array_shape)
            array_index = array_start + (index - start) / step * array_step
            slices.append(slice(array_index, array_index+1, 1))

        return value[slices].squeeze()

    def getimage(self, index, value):
        return value[index][self.array_slices[1:]]

    def __repr__(self):
        return "PaddedBlockGroupingKey" \
               "\nshape: %s\nslices: %s\npadded_slices: %s\narray_shape: %s\narray_slices: %s)" % \
               (self.shape, self.padded_slices, self.slices, self.array_shape, self.array_slices)
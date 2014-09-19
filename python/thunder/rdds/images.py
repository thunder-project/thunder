import glob
import os
from numpy import ndarray, fromfile, int16, uint16, prod, concatenate, reshape, zeros
import itertools
import struct
import cStringIO as StringIO
from matplotlib.pyplot import imread
import sys
from series import Series
from thunder.rdds.data import Data


class Images(Data):

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
        this method will return a new ImageBlocks with n*z items, one for each z-plane in the passed
        images. There will be z unique keys, (0, 0, 0), (0, 0, 1), ... (0, 0, z-1). Each value will be
        an instance of ImageBlockValue, representing an n,z plane within a larger volume of dimensions
        n,x,y,z.

        This method is not expected to be called directly by end users.

        Parameters:
        -----------
        groupingdim: integer, -ndim <= groupingdim < ndim, where ndim is the dimensionality of the image
            Specifies the index of the dimension along which the images are to be divided into planes.
            Negative groupingdims are interpreted as counting from the end of the sequence of dimensions,
            so groupingdim == -1 represents slicing along the last dimension. -1 is the default.

        Todos:
        ------
        * support concatinating time points as final dimension rather than first - would allow for contiguous
        array access for fortran-ordered data
        * figure out some way for total image number to be evaluated lazily

        """
        ndim = len(self.dims)
        _dim = groupingDim if groupingDim >= 0 else ndim + groupingDim
        if _dim < 0 or _dim >= ndim:
            raise ValueError("Dimension to group by (%d) must be less than array dimensionality (%d)" %
                             (_dim, ndim))

        totnumimages = self.nimages

        def _groupByPlanes(imagearyval, _groupingdim, _tp, _numtp):
            ret_vals = []
            origndim = imagearyval.ndim
            origimagearyshape = imagearyval.shape
            for groupingidx in xrange(origimagearyshape[_groupingdim]):
                blockval = ImageBlockValue.fromArrayByPlane(imagearyval, planedim=_groupingdim, planeidx=groupingidx,
                                                            docopy=False)
                blockval = blockval.addDimension(newdimidx=_tp, newdimsize=_numtp)
                newkey = [0] * origndim
                newkey[_groupingdim] = groupingidx
                ret_vals.append((tuple(newkey), blockval))
            return ret_vals

        def _groupByPlanesAdapter(keyval):
            tpkey, imgaryval = keyval
            return _groupByPlanes(imgaryval, _dim, tpkey, totnumimages)

        return ImageBlocks(self.rdd.flatMap(_groupByPlanesAdapter, preservesPartitioning=False))

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

    def toSeries(self, groupingdim=None):

        # normalize grouping dimension, or get a reasonable grouping dimension if unspecified
        # this may trigger a first() call:
        gd = self.__validateOrCalcGroupingDim(groupingDim=groupingdim)

        # returns keys of (z, y, x); with z as grouping dimension, key values will be (0, 0, 0), (1, 0, 0), ...
        # (z-1, 0, 0)
        blocksdata = self._toBlocksByImagePlanes(groupingDim=gd)
        # key is still spatial without time; e.g. (z, y, x)
        # block orig shape and slices include time and spatial dimensions; (t, z, y, x)
        # toSeries expects grouping dimension to be relative to higher-dimensional space, including time:
        expandedgd = gd + 1
        return blocksdata.toSeries(expandedgd, seriesDim=0)

    def saveAsBinarySeries(self, outputdirname, groupingdim=None, overwrite=False):
        if overwrite and os.path.isdir(outputdirname):
            import shutil
            shutil.rmtree(outputdirname)
        os.mkdir(outputdirname)

        gd = self.__validateOrCalcGroupingDim(groupingDim=groupingdim)
        blocksdata = self._toBlocksByImagePlanes(groupingDim=gd)

        expandedgd = gd + 1
        binseriesrdd = blocksdata.toBinarySeries(groupingDim=expandedgd, seriesDim=0)

        def writeBinarySeriesFile(kv):
            binlabel, binvals = kv
            with open(os.path.join(outputdirname, binlabel + ".bin"), 'wb') as f_:
                f_.write(binvals)

        binseriesrdd.foreach(writeBinarySeriesFile)

        # write configuration file
        conf = {'input': outputdirname, 'dims': self.dims,
                'nkeys': len(self.dims), 'nvalues': self.nimages,
                'format': self.dtype, 'keyformat': 'int16'}
        with open(os.path.join(outputdirname, "conf.json"), 'w') as fconf:
            import json
            json.dump(conf, fconf, indent=2)

        # touch "SUCCESS" file as final action
        with open(os.path.join(outputdirname, "SUCCESS"), 'w'):
            pass


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

        def reader(filepath):
            with open(filepath, 'rb') as f:
                stack = fromfile(f, int16, prod(dims)).reshape(dims, order='F')
            return stack.astype(uint16)

        files = self.listFiles(datafile, ext=ext, startidx=startidx, stopidx=stopidx)
        return Images(self._loadFiles(files, reader), nimages=len(files), dims=dims, dtype='uint16')

    def fromTif(self, datafile, ext='tif', startidx=None, stopidx=None):

        return self.fromFile(datafile, imread, ext=ext, startidx=startidx, stopidx=stopidx)

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

        def multitifReader(f):
            multipage = Image.open(f)
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
            return concatenate(imgarys, axis=2)

        return self.fromFile(datafile, multitifReader, ext=ext, startidx=startidx, stopidx=stopidx)

    def _loadFiles(self, files, readerfcn):
        return self.sc.parallelize(enumerate(files), len(files)).map(lambda (k, v): (k, readerfcn(v)))

    def fromFile(self, datafile, reader, ext=None, startidx=None, stopidx=None):
        files = self.listFiles(datafile, ext=ext, startidx=startidx, stopidx=stopidx)
        return Images(self._loadFiles(files, reader), nimages=len(files))

    def fromPng(self, datafile, ext='png', startidx=None, stopidx=None):
        return self.fromFile(datafile, imread, ext=ext, startidx=startidx, stopidx=stopidx)

    @staticmethod
    def listFiles(datafile, ext=None, startidx=None, stopidx=None):

        if os.path.isdir(datafile):
            if ext:
                files = sorted(glob.glob(os.path.join(datafile, '*.' + ext)))
            else:
                files = sorted(os.listdir(datafile))
        else:
            files = sorted(glob.glob(datafile))

        if len(files) < 1:
            raise IOError('cannot find files of type "%s" in %s' % (ext if ext else '*', datafile))

        if startidx or stopidx:
            if startidx is None:
                startidx = 0
            if stopidx is None:
                stopidx = len(files)
            files = files[startidx:stopidx]

        return files


class ImageBlocks(Data):

    @staticmethod
    def __validateGroupingAndSeriesDims(groupingDim, seriesDim):
        if groupingDim == seriesDim:
            raise ValueError("Dimension used to collect image blocks (%d) cannot be the same as " % groupingDim +
                             "time series dimension (%d)" % seriesDim)

    @staticmethod
    def _blockToSeries(blockKey, blockVal, keyGroupingDim, seriesDim):
        planeIdx = blockKey[keyGroupingDim]
        for seriesKey, seriesVal in blockVal.toSeriesIter(seriesDim=seriesDim):
            # add plane index back into key:
            seriesKey = list(seriesKey)
            seriesKey.insert(keyGroupingDim, planeIdx)
            yield tuple(seriesKey), seriesVal

    def toSeries(self, groupingDim=1, seriesDim=0):
        """
        key is expected to be spatial, without time; (z, y, x)
        block origshape and slices have time appended; (t, z, y, x)
        groupingDim is expected to be relative to block value, in higher-dimensional space (including time; t, z, y, x)
        seriesDim is expected to be relative to block value, in higher-dimensional space (including time; t, z, y, x)
        """
        self.__validateGroupingAndSeriesDims(groupingDim, seriesDim)

        def blockToSeriesAdapter(kv):
            blockKey, blockVal = kv
            return ImageBlocks._blockToSeries(blockKey, blockVal, keyGroupingDim, seriesDim)

        blockedrdd = self._groupIntoSeriesBlocks(groupingDim=groupingDim)

        # the key does not include the time dimension, but the groupingDim argument is passed
        #   assuming that time is included. decrement this if needed for reference into the key.
        keyGroupingDim = groupingDim - 1 if seriesDim < groupingDim else groupingDim

        # convert resulting (z, y, x) keys and (t, y, x) blocks into multiple
        #   (z, y, x) keys (one per voxel) and t arrays
        # returns generator of (z, y, x) array data for all z, y, x
        seriesrdd = blockedrdd.flatMap(blockToSeriesAdapter)
        return Series(seriesrdd)

    def toBinarySeries(self, groupingDim=1, seriesDim=0):
        self.__validateGroupingAndSeriesDims(groupingDim, seriesDim)

        blockedrdd = self._groupIntoSeriesBlocks(groupingDim=groupingDim)

        # the key does not include the time dimension, but the groupingDim argument is passed
        #   assuming that time is included. decrement this if needed for reference into the key.
        keyGroupingDim = groupingDim - 1 if seriesDim < groupingDim else groupingDim

        def blockToBinarySeries(kv):
            blockKey, blockVal = kv
            label = '-'.join("%05g" % k for k in blockKey)
            keypacker = None
            buf = StringIO.StringIO()
            for seriesKey, series in ImageBlocks._blockToSeries(blockKey, blockVal, keyGroupingDim, seriesDim):
                if keypacker is None:
                    keypacker = struct.Struct('h'*len(seriesKey))
                # print >> sys.stderr, seriesKey, series, series.tostring().encode('hex')
                buf.write(keypacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return label, val

        return blockedrdd.map(blockToBinarySeries)

    def _groupIntoSeriesBlocks(self, groupingDim=1):
        """Combine blocks representing individual image planes into a single planes-by-time volume

        Returns:
        --------
        RDD, key/value: tuple of int, ImageBlockValue
        key:
            spatial indicies of start of block, for instance (z, y, x): (0, 0, 0), (1, 0, 0),... (z_max-1, 0, 0)
        value:
            ImageBlockValue with single fully-populated array, dimensions of time by space, for instance (t, y, x):
            ary[0:t_max, :, :]
        """
        # squeeze out dimension used for grouping
        squeezedData = self.squeeze(groupingDim)

        # combine blocks representing individual image planes into a single planes-by-time volume:
        # key will be (z, y, x) for start of block
        # val will be single blockValue array data with dimensions for instance (t, y, x)
        return squeezedData.rdd.groupByKey().mapValues(ImageBlockValue.fromBlocks)

    def squeeze(self, squeezedimidx=-1):
        """Removes the specified dimension from all blocks.

        Keys remain unchanged.
        """
        return ImageBlocks(self.rdd.mapValues(lambda v: v.squeeze(squeezedimidx)))


class ImageBlockValue(object):
    """
    Helper data structure for transformations from Images to Series.

    Not intended for direct use by clients.
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
        slices[planedim] = slice(planeidx, planeidx+1)
        if docopy:
            return cls(imagearray.shape, tuple(slices), imagearray[slices].copy())
        else:
            return cls(imagearray.shape, tuple(slices), imagearray[slices])

    @classmethod
    def fromBlocks(cls, blocksIter):
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
        newslices.insert(0, slice(newdimidx, newdimidx+1))
        newblockshape = list(self.values.shape)
        newblockshape.insert(0, 1)
        newvalues = reshape(self.values, tuple(newblockshape))
        return type(self)(tuple(newshape), tuple(newslices), newvalues)

    def squeeze(self, squeezedimidx=-1):
        """Returns a new ImageBlockValue with the specified singleton dimension removed.

        Passed dimension will be removed from origslices, origshape, and values. Exception will
        be thrown (by numpy) if the dimension is not a singleton in values.

        See numpy.squeeze.
        """
        newvalues = self.values.squeeze(squeezedimidx)
        newslices = list(self.origslices)
        del newslices[squeezedimidx]
        newshape = list(self.origshape)
        del newshape[squeezedimidx]

        return ImageBlockValue(tuple(newshape), tuple(newslices), newvalues)

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
        for idxSeq in itertools.product(*rangeiters):
            slices = [slice(idx, idx+1) for idx in idxSeq]
            slices.insert(insertDim, slice(None))
            series = self.values[slices].squeeze()
            yield tuple(idxSeq), series

    def _get_range_iterators(self):
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
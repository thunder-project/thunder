"""Provides ImagesLoader object and helpers, used to read Images data from disk or other filesystems.
"""
from matplotlib.pyplot import imread
from io import BytesIO
from numpy import dstack, frombuffer, ndarray, prod
from thunder.rdds.fileio.readers import getParallelReaderForPath
from thunder.rdds.images import Images


class ImagesLoader(object):
    """Loader object used to instantiate Images data stored in a variety of formats.
    """
    def __init__(self, sparkcontext):
        """Initialize a new ImagesLoader object.

        Parameters
        ----------
        sparkcontext: SparkContext
            The pyspark SparkContext object used by the current Thunder environment.
        """
        self.sc = sparkcontext

    def fromArrays(self, arrays):
        """Load Images data from passed sequence of numpy arrays.

        Expected usage is mainly in testing - having a full dataset loaded in memory
        on the driver is likely prohibitive in the use cases for which Thunder is intended.
        """
        # if passed a single array, cast it to a sequence of length 1
        if isinstance(arrays, ndarray):
            arrays = [arrays]

        dims = None
        dtype = None
        for ary in arrays:
            if dims is None:
                dims = ary.shape
                dtype = ary.dtype
            if not ary.shape == dims:
                raise ValueError("Arrays must all be of same shape; got both %s and %s" %
                                 (str(dims), str(ary.shape)))
            if not ary.dtype == dtype:
                raise ValueError("Arrays must all be of same data type; got both %s and %s" %
                                 (str(dtype), str(ary.dtype)))

        return Images(self.sc.parallelize(enumerate(arrays), len(arrays)),
                      dims=dims, dtype=str(dtype), nimages=len(arrays))

    def fromStack(self, datapath, dims, ext='stack', startidx=None, stopidx=None):
        """Load an Images object stored in a directory of flat binary files

        The RDD wrapped by the returned Images object will have a number of partitions equal to the number of image data
        files read in by this method.

        Currently all binary data read by this method is assumed to be formatted as signed 16 bit integers in native
        byte order.

        Parameters
        ----------

        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename.

        dims: tuple of positive int
            Dimensions of input image data, similar to a numpy 'shape' parameter.

        ext: string, optional, default "stack"
            Extension required on data files to be loaded.

        startidx, stopidx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `datapath` and `ext`. Interpreted according to python slice indexing conventions.
        """
        if not dims:
            raise ValueError("Image dimensions must be specified if loading from binary stack data")

        def toArray(buf):
            # previously we were casting to uint16 - still necessary?
            return frombuffer(buf, dtype='int16', count=prod(dims)).reshape(dims, order='C')

        reader = getParallelReaderForPath(datapath)(self.sc)
        readerrdd = reader.read(datapath, ext=ext, startidx=startidx, stopidx=stopidx)
        return Images(readerrdd.mapValues(toArray), nimages=reader.lastnrecs, dims=dims, dtype='int16')

    def fromTif(self, datapath, ext='tif', startidx=None, stopidx=None):
        """Load an Images object stored in a directory of (single-page) tif files

        The RDD wrapped by the returned Images object will have a number of partitions equal to the number of image data
        files read in by this method.

        Parameters
        ----------

        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename.

        ext: string, optional, default "tif"
            Extension required on data files to be loaded.

        startidx, stopidx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `datapath` and `ext`. Interpreted according to python slice indexing conventions.
        """
        def readTifFromBuf(buf):
            fbuf = BytesIO(buf)
            return imread(fbuf, format='tif')

        reader = getParallelReaderForPath(datapath)(self.sc)
        readerrdd = reader.read(datapath, ext=ext, startidx=startidx, stopidx=stopidx)
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
            Image = None
            raise ImportError("fromMultipageTif requires a successful 'from PIL import Image'; " +
                              "the PIL/pillow library appears to be missing or broken.", e)
        from thunder.utils.common import pil_to_array

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
        """Load an Images object stored in a directory of png files

        The RDD wrapped by the returned Images object will have a number of partitions equal to the number of image data
        files read in by this method.

        Parameters
        ----------

        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename.

        ext: string, optional, default "png"
            Extension required on data files to be loaded.

        startidx, stopidx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `datapath` and `ext`. Interpreted according to python slice indexing conventions.
        """
        def readPngFromBuf(buf):
            fbuf = BytesIO(buf)
            return imread(fbuf, format='png')

        reader = getParallelReaderForPath(datafile)(self.sc)
        readerrdd = reader.read(datafile, ext=ext, startidx=startidx, stopidx=stopidx)
        return Images(readerrdd.mapValues(readPngFromBuf), nimages=reader.lastnrecs)
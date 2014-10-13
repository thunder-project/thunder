from matplotlib.pyplot import imread
from io import BytesIO
from numpy import frombuffer, prod, dstack
from thunder.rdds.fileio.readers import getParallelReaderForPath
from thunder.rdds.images import Images


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
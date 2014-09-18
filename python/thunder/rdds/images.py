import glob
import os
from numpy import ndarray, fromfile, int16, uint16, prod, concatenate
from matplotlib.pyplot import imread
from thunder.rdds.data import Data


class Images(Data):

    def __init__(self, rdd, dims=None):
        super(Images, self).__init__(rdd)
        self._dims = dims

    @property
    def dims(self):
        if self._dims is None:
            record = self.rdd.first()
            self._dims = record[1].shape
        return self._dims

    @staticmethod
    def _check_type(record):
        if not isinstance(record[0], tuple):
            raise Exception('Keys must be tuples')
        if not isinstance(record[1], ndarray):
            raise Exception('Values must be ndarrays')

    def toSeries(self):

        raise NotImplementedError

    def toBlocks(self):

        raise NotImplementedError

    def saveAsBinarySeries(self):

        blocks = self.toBlocks()


class ImagesLoader(object):

    def __init__(self, sparkcontext, filerange=None):
        self.sc = sparkcontext
        self.filerange = filerange

    def fromStack(self, datafile, dims, ext='stack'):

        def reader(filepath):
            f = open(filepath, 'rb')
            stack = fromfile(f, int16, prod(dims)).reshape(dims, order='F')
            f.close()
            return stack.astype(uint16)

        return Images(self.fromFile(datafile, reader, ext=ext), dims=dims)

    def fromTif(self, datafile, ext='tif'):

        return Images(self.fromFile(datafile, imread, ext=ext))

    def fromMultipageTif(self, datafile, ext='tif'):
        """
        Sets up a new Images object with data to be read from one or more multi-page tif files.

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

        files = self.listFiles(datafile, ext)
        rdd = self.sc.parallelize(enumerate(files), len(files)).map(lambda (k, v): (k, multitifReader(v)))
        return Images(rdd)

    def fromFile(self, datafile, reader, ext=None):

        files = self.listFiles(datafile, ext=ext)
        return self.sc.parallelize(enumerate(files), len(files)).map(lambda (k, v): (k, reader(v)))

    def fromPng(self, datafile, ext='png'):

        return Images(self.fromFile(datafile, imread, ext=ext))

    def listFiles(self, datafile, ext=None):

        if os.path.isdir(datafile):
            if ext:
                files = sorted(glob.glob(os.path.join(datafile, '*.' + ext)))
            else:
                files = sorted(os.listdir(datafile))
        else:
            files = sorted(glob.glob(datafile))

        if len(files) < 1:
            raise IOError('cannot find files of type "%s" in %s' % (ext if ext else '*', datafile))

        if self.filerange:
            files = files[self.filerange[0]:self.filerange[1]+1]

        return files

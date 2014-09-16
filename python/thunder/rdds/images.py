import glob
import os
from numpy import shape, ndarray, fromfile, int16, uint16, prod
from matplotlib.pyplot import imread
from thunder.rdds.data import Data


class Images(Data):

    def __init__(self, rdd, dims=None):
        super(Images, self).__init__(rdd)
        if dims is None:
            record = self._rdd.first()
            self.dims = shape(record)
        else:
            self.dims = dims

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

    def saveAsSeries(self):

        blocks = self.toBlocks()



class ImagesLoader(object):

    def __init__(self, dims=None, valuetype=None, filerange=None):
        self.dims = dims
        self.valuetype = valuetype
        self.filerange = filerange

    def fromStack(self, datafile, sc):

        def reader(file):
            f = open(file, 'rb')
            stack = fromfile(f, int16, prod(self.dims)).reshape(self.dims, order='F')
            f.close()
            return stack.astype(uint16)

        return Images(self.fromFile(datafile, sc, reader, ext='stack'), dims=self.dims)

    def fromTif(self, datafile, sc):

        def reader(file):
            return imread(file)

        return Images(self.fromFile(datafile, sc, reader, ext='tif'))

    def fromPng(self, datafile, sc):

        def reader(file):
            return imread(file)

        return Images(self.fromFile(datafile, sc, reader, ext='png'))

    def fromFile(self, datafile, reader, sc, ext):

        files = self.listFiles(datafile, ext)
        files = zip(range(0, len(files)), files)
        return sc.parallelize(files, len(files)).map(lambda (k, v): (k, reader(v)))

    def listFiles(self, datafile, ext):

        if os.path.isdir(datafile):
            files = sorted(glob.glob(os.path.join(datafile, '*.' + ext)))
        else:
            files = sorted(glob.glob(datafile))

        if len(files) < 1:
            raise IOError('cannot find files of type %s in %s' % (ext, datafile))

        if self.filerange:
            files = files[self.filerange[0]:self.filerange[1]+1]

        return files
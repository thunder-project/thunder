""" Class and methods for transforming data types """

import os
import glob
from numpy import prod, zeros, fromfile, int16, ceil, savetxt


#def transform(sc, path, npartitions):

    # try to figure out the format
    # if stack -> data = Transform(sc, npartitions).fromstack(path)
    # if tif -> data = Transform(sc, npartitions).fromtif(path)

    # if given a path and a save dir, make sure we're given a format
    # if binary -> data.tobinary(savedir)
    # if text -> data.totext(savedir)

    # otherwise, just return the rdd
    # return data.tordd()


class Transform(object):

    def __init__(self, sc, npartitions=None):
        if npartitions is not None:
            self.npartitions = npartitions
        else:
            self.npartitions = sc.defaultMinPartitions
        self._sc = sc

    def fromstack(self, path, dims):
        """ Transform data from binary stack files,
        assuming raw data are a series of stack files

        (stack, file 0), (stack, file 1), ... (stack, file n)

        splits each stack into k blocks and constructs an RDD
        where each record is an ndarray with the elements of a
        single block across the rows and the files as columns, i.e.

        (block 0, file 0...n), (block 1, file 0...n), ... (block k, file 0...n)

        where a block is a contiguous region from a stack
        """

        def readblock(part, files):
            position = part * blocksize
            if (position + blocksize) > totaldim:
                mat = zeros((totaldim - position, len(files)))
            else:
                mat = zeros((blocksize, len(files)))
            for i, f in enumerate(files):
                fid = open(f, "rb")
                fid.seek(position * 2)
                mat[:, i] = fromfile(fid, dtype=int16, count=blocksize)
            return mat

        # get the paths to the data
        files = glob.glob(path)
        self.files = files

        # get the total stack dimensions
        totaldim = float(prod(dims))

        # compute a block size
        blocksize = int(ceil(int(ceil(totaldim / float(self.npartitions)))))

        # recompute number of partitions based on block size
        self.npartitions = int(round(totaldim / float(blocksize)))

        # map over parts
        parts = range(0, self.npartitions)
        rdd = self._sc.parallelize(parts, len(parts)).map(lambda ip: (ip, readblock(ip, files)))

        self._rdd = rdd
        return self

    def tobinary(self, path):
        """ Write data to flat binary files """

        def writeblock(part, mat, path):
            filename = path + str(part) + ".bin"
            mat.tofile(filename)

        self._rdd.foreach(lambda (ip, mat): writeblock(ip, mat, path))

    def totext(self, path):
        """ Write data to text files """

        #def writeblock(part, mat, path):
        #    filename = path + "part-" + str(part) + ".txt"
        #    savetxt(filename, mat, fmt="%g")

        #self._rdd.foreach(lambda (ip, mat): writeblock(ip, mat, path))
        n = len(self.files)
        self._rdd.values().map(lambda x: "\n".join(map(lambda y: ("%g " * n) % tuple(y), x))).saveAsTextFile(path)

    def tordd(self):
        return self._rdd.flatMap(lambda (k, x): list(x))

""" Class and methods for transforming data types """

import os
import glob
from numpy import prod, zeros, fromfile, int16, ceil, shape, asarray, concatenate
from thunder.utils import indtosub


def transform(sc, datadir, npartitions=None, savedir=None, saveformat=None, dims=None):
    """ Convienience function for transforming input formats """

    # find first file in the path
    if os.path.isdir(datadir):
        files = sorted(glob.glob(os.path.join(datadir, '*')))
    else:
        files = sorted(glob.glob(datadir))

    # determine format
    loadformat = os.path.split(files[0])[1].split('.')[1]

    if loadformat == 'stack':
        if not dims:
            raise StandardError("must provide dimensions for stack data")
        data = Transform(sc, npartitions).fromstack(datadir, dims)
    elif loadformat == 'tif':
        data = Transform(sc, npartitions).fromtif(datadir)
    else:
        raise NotImplementedError("loading from %s not implemented" % loadformat)

    if saveformat:
        if saveformat == 'binary':
            data.tobinary(savedir)
        elif saveformat == 'text':
            data.totext(savedir)
        else:
            raise NotImplementedError("saving to %s not implemented" % saveformat)
    else:
        return data.tordd()


class Transform(object):

    def __init__(self, sc, npartitions=None):
        if npartitions is None:
            self.npartitions = sc.defaultMinPartitions
        else:
            self.npartitions = npartitions
        self._sc = sc

    def fromstack(self, datadir, dims):
        """ Transform data from binary multi-band stack files,
        assuming raw data are a series of stack files with int16 values

        (stack, file 0), (stack, file 1), ... (stack, file n)

        splits each stack into k blocks and constructs an RDD
        where each record is an ndarray with the elements of a
        single block across the rows and the files as columns, i.e.

        (block 0, file 0...n), (block 1, file 0...n), ... (block k, file 0...n)

        where a block is a contiguous region from a stack
        """

        # TODO currently assumes int16, allow for arbitrary formats

        def readblock(part, files, blocksize):
            # get start position for this block
            position = part * blocksize

            # adjust if at end of file
            if (position + blocksize) > totaldim:
                blocksize = int(totaldim - position)

            # loop over files, loading one block from each
            mat = zeros((blocksize, len(files)))

            for i, f in enumerate(files):
                fid = open(f, "rb")
                fid.seek(position * 2)
                mat[:, i] = fromfile(fid, dtype=int16, count=blocksize)

            # append subscript keys based on dimensions
            lininds = range(position + 1, position + shape(mat)[0] + 1)
            keys = asarray(map(lambda (k, v): k, indtosub(zip(lininds, zeros(blocksize)), dims)))
            partlabel = "%05g-" % part + (('%05g-' * len(dims)) % tuple(keys[0]))[:-1]
            return partlabel, concatenate((keys, mat), axis=1).astype(int16)

        # get the paths to the data
        if os.path.isdir(datadir):
            files = sorted(glob.glob(os.path.join(datadir, '*.stack')))
        else:
            files = sorted(glob.glob(datadir))

        if len(files) < 1:
            raise IOError('cannot find any files in %s' % datadir)

        self.files = files
        self.nkeys = len(dims)
        self.nvals = len(files)

        # get the total stack dimensions
        totaldim = float(prod(dims))

        # compute a block size
        blocksize = int(ceil(int(ceil(totaldim / float(self.npartitions)))))

        # recompute number of partitions based on block size
        self.npartitions = int(round(totaldim / float(blocksize)))

        # map over parts
        parts = range(0, self.npartitions)
        rdd = self._sc.parallelize(parts, len(parts)).map(lambda ip: readblock(ip, files, blocksize))
        self._rdd = rdd

        return self

    def fromtif(self, path):
        """ Transform data from tif image stacks """

        raise NotImplementedError('loading from tifs not yet implemented')

    def tobinary(self, path):
        """ Write blocks to flat binary files """

        def writeblock(part, mat, path):
            filename = os.path.join(path, "part-" + str(part) + ".bin")
            mat.tofile(filename)

        if os.path.isdir(path):
            raise IOError('path %s already exists' % path)
        else:
            os.mkdir(path)

        self._rdd.foreach(lambda (ip, mat): writeblock(ip, mat, path))

    def totext(self, path):
        """ Write blocks to text files """

        n = self.nkeys + self.nvals
        self._rdd.values().map(lambda x: "\n".join(map(lambda y: ("%g " * n) % tuple(y), x))).saveAsTextFile(path)

    def tordd(self):
        """ Convert blocks to RDD of key-value records """

        n = self.nkeys
        return self._rdd.values().flatMap(lambda x: list(x)).map(lambda x: (tuple(x[0:n].astype(int)), x[n:]))

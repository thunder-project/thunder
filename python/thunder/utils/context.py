""" Simple wrapper for a Spark Context to provide loading functionality """

import glob
import os
from numpy import int16, dtype, frombuffer, zeros, fromfile, asarray, ceil, shape, concatenate, prod
from pyspark import SparkContext
from thunder.utils.load import PreProcessor, Parser, indtosub
from thunder.utils import DataSets


class ThunderContext():
    """
    Wrapper for a Spark Context

    Attributes
    ----------
    `_sc` : SparkContext
        Spark context for Spark functionality
    """

    def __init__(self, sparkcontext):
        self._sc = sparkcontext

    @classmethod
    def start(cls, *args, **kwargs):
        """Starts a ThunderContext using the same arguments as SparkContext"""
        return ThunderContext(SparkContext(*args, **kwargs))

    def loadText(self, datafile, nkeys=3, filter=None, npartitions=None):
        """
        Load data from text file (or a directory of files) with rows as
        <k1> <k2> ... <v1> <v2> ...
        where <k1> <k2> ... are keys and <v1> <v2> ... are the data values
        and rows are separated by line breaks
        Files can be local or stored on HDFS / S3
        If passed a directory, will automatically sort files by name

        Parameters
        ----------
        datafile : str
            Location of file(s)

        nkeys : int, optional, default = 3
            Number of keys per data point

        filter : str, optional, default = None (no preprocessing)
            Which preprocessing to perform

        npartitions : int, optional, default = None
            Number of partitions

        Returns
        -------
        data : RDD of (tuple, array) pairs
            The parsed and preprocessed data as an RDD
        """

        if os.path.isdir(datafile):
            files = sorted(glob.glob(os.path.join(datafile, '*.txt')))
            datafile = ''.join([files[x] + ',' for x in range(0, len(files))])[0:-1]

        lines = self._sc.textFile(datafile, npartitions)
        parser = Parser(nkeys)
        data = lines.map(parser.get)

        return preprocess(data, method=filter)

    def loadBinary(self, datafile, nvalues, nkeys, format=int16, filter=None):
        """
        Load data from flat binary file (or a directory of files) with format
        <k1> <k2> ... <v1> <v2> ... <k1> <k2> ... <v1> <v2> ...
        where <k1> <k2> ... are keys and <t1> <t2> ... are data values
        Each record must contain the same total number of keys and values
        Files can be local or stored on HDFS / S3

        Parameters
        ----------
        datafile : str
            Location of file(s)

        nvalues : int
            Number of values per record

        nkeys : int
            Number of keys per record

        format : numpy.dtype, optional, default = int16
            Format to use when parsing binary data

        filter : str, optional, default = None (no preprocessing)
            Which preprocessing to perform

        Returns
        -------
        data : RDD of (tuple, array) pairs
            The parsed and preprocessed data as an RDD
        """

        nvalues += nkeys
        nvalues *= dtype(format).itemsize
        lines = self._sc.newAPIHadoopFile(datafile, 'thunder.util.io.hadoop.FixedLengthBinaryInputFormat',
                                          'org.apache.hadoop.io.LongWritable',
                                          'org.apache.hadoop.io.BytesWritable',
                                          conf={'recordLength': str(nvalues)})

        data = lines.map(lambda (k, v): frombuffer(v, format))\
            .map(lambda v: (tuple(v[0:nkeys].astype(int)), v[nkeys:].astype(float)))

        return preprocess(data, method=filter)

    def makeExample(self, dataset, **opts):
        """
        Make an example data set for testing analyses
        see DataSets

        Parameters
        ----------
        dataset : str
            Which dataset to generate

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset
        """

        return DataSets.make(self._sc, dataset, **opts)

    def loadExample(self, dataset):
        """
        Load a local example data set for testing analyses

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset
        """

        path = os.path.dirname(os.path.realpath(__file__))

        if dataset == "iris":
            return self.loadText(os.path.join(path, 'data/iris.txt'))
        elif dataset == "fish":
            return self.loadText(os.path.join(path, 'data/fish.txt'))
        else:
            raise NotImplementedError("dataset '%s' not found" % dataset)

    def loadExampleEC2(self, dataset):
        """
        Load an example data set from EC2

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset
        """

        if 'ec' not in self._sc.master:
            raise Exception("must be running on EC2 to load this example data sets")
        elif dataset == "zebrafish-optomotor-response":
            path = 's3n://zebrafish.datasets/optomotor-response/1/'
            return self.loadText(path + 'data/dat_plane*.txt', npartitions=1000)
        else:
            raise NotImplementedError("dataset '%s' not availiable" % dataset)

    def convertStack(self, datafile, dims, npartitions=None, savefile=None):
        """
        Convert data from binary stack files to reformatted flat binary files,
        see also convertStack

        Parameters
        ----------
        datafile : str
            File(s) or directory to convert

        dims : list
            Stack dimensions

        npartitions : int, optional, automatically set
            Number of partitions for splitting data

        savefile : str, optional, default = None (directly return RDD)
            Location to save the converted data
        """
        rdd = self.importStackAsBlocks(datafile, dims, npartitions=npartitions)

        # save blocks of data to flat binary files
        def writeblock(part, mat, path):
            filename = os.path.join(path, "part-" + str(part) + ".bin")
            mat.tofile(filename)

        if os.path.isdir(savefile):
            raise IOError('path %s already exists' % savefile)
        else:
            os.mkdir(savefile)

        rdd.foreach(lambda (ip, mat): writeblock(ip, mat, savefile))

    def importStack(self, datafile, dims, npartitions=None, filter=None):
        """
        Import data from binary stack files as an RDD, see also convertStack

        Parameters
        ----------
        datafile : str
            File(s) or directory to import

        dims : list
            Stack dimensions

        npartitions : int, optional, automatically set
            Number of partitions for splitting data

        filter : str, optional, default = None (no preprocessing)
            Which preprocessing to perform

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Parsed and preprocessed data
        """
        rdd = self.importStackAsBlocks(datafile, dims, npartitions=npartitions)
        nkeys = len(dims)
        data = rdd.values().flatMap(lambda x: list(x)).map(lambda x: (tuple(x[0:nkeys].astype(int)), x[nkeys:]))
        return preprocess(data, method=filter)

    def importStackAsBlocks(self, datafile, dims, npartitions=None):
        """
        Convert data from binary stack files to blocks of an RDD,
        which can either be saved to flat binary files,
        or returned as an flattened RDD (see convertStack and importStack)
        """

        # get the paths to the data
        if os.path.isdir(datafile):
            files = sorted(glob.glob(os.path.join(datafile, '*.stack')))
        else:
            files = sorted(glob.glob(datafile))
        if len(files) < 1:
            raise IOError('cannot find any files in %s' % datafile)

        # get the total stack dimensions
        totaldim = float(prod(dims))

        # compute a block size
        # TODO add automatic calculation of evenly distributed partitions
        blocksize = int(ceil(int(ceil(totaldim / float(npartitions)))))

        # recompute number of partitions based on block size
        npartitions = int(round(totaldim / float(blocksize)))

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

        # map over parts
        parts = range(0, npartitions)
        return self._sc.parallelize(parts, len(parts)).map(lambda ip: readblock(ip, files, blocksize))

    def loadBinaryLocal(self, datafile, nvalues, nkeys, format=int16, keyfile=None, method=None):
        """
        Load data from a local binary file
        """

        raise NotImplementedError

    def loadMatLocal(self, datafile, keyfile=None, method=None):
        """
        Load data from a local MAT file
        """

        raise NotImplementedError


def preprocess(data, method=None):

    if method:
        preprocessor = PreProcessor(method)
        return data.mapValues(preprocessor.get)
    else:
        return data


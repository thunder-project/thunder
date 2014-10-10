""" Simple wrapper for a Spark Context to provide loading functionality """

import os
import json
from numpy import asarray, floor, ceil
from thunder.rdds.fileio.imagesloader import ImagesLoader
from thunder.rdds.fileio.seriesloader import SeriesLoader
from thunder.utils.datasets import DataSets
from thunder.utils.common import checkparams


class ThunderContext():
    """
    Wrapper for a SparkContext that provides functionality for loading data.

    Also supports creation of example datasets, and loading example
    data both locally and from EC2.

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
        from pyspark import SparkContext
        return ThunderContext(SparkContext(*args, **kwargs))

    def loadSeries(self, datafile, nkeys=None, nvalues=None, inputformat='binary', minPartitions=None,
                   conffile='conf.json', keytype=None, valuetype=None):
        """
        Loads a Series object from data stored as text or binary files.

        Supports single files or multiple files stored on a local file system, a networked file system (mounted
        and available on all cluster nodes),Amazon S3, or HDFS.

        Parameters
        ----------
        datafile: string
            path to single file or directory. If directory, will be expected to contain multiple *.txt (if
            text) or *.bin (if binary) data files.

        nkeys: int, optional
            dimensionality of data keys. (For instance, (x,y,z) keyed data for 3-dimensional image timeseries data.)
            For text data, number of keys must be specified in this parameter; for binary data, number of keys must be
            specified either in this parameter or in a configuration file named by the 'conffile' argument if this
            parameter is not set.

        nvalues: int, optional
            Number of values expected to be read. For binary data, nvalues must be specified either in this parameter
            or in a configuration file named by the 'conffile' argument if this parameter is not set.

        inputformat: string, optional, default = 'binary'
            Format of data to be read. Must be either 'text' or 'binary'.

        minPartitions: int, optional
            Explicitly specify minimum number of Spark partitions to be generated from this data. Used only for
            text data. Default is to use minParallelism attribute of Spark context object.

        conffile: string, optional, default 'conf.json'
            Path to JSON file with configuration options including 'nkeys' and 'nvalues'. If a file is not found at the
            given path, then the base directory given in 'datafile' will also be checked. Parameters specified as
            explicit arguments to this method take priority over those found in conffile if both are present.
        """
        checkparams(inputformat, ['text', 'binary'])

        loader = SeriesLoader(self._sc, minPartitions=minPartitions)

        if inputformat.lower() == 'text':
            data = loader.fromText(datafile, nkeys=nkeys)
        else:
            # must be either 'text' or 'binary'
            data = loader.fromBinary(datafile, conffilename=conffile, nkeys=nkeys, nvalues=nvalues,
                                     keytype=keytype, valuetype=valuetype)

        return data

    def loadImages(self, datafile, dims=None, inputformat='stack', startidx=None, stopidx=None):
        """
        Loads an Images object from data stored as a binary image stack, tif, tif-stack, or png files.

        Supports single files or multiple files, stored on a local file system, a networked file sytem
        (mounted and available on all nodes), or Amazon S3

        Parameters
        ----------
        datafile: string
            path to single file or directory. If directory, will be expected to contain multiple *.stack, *.tif, or
            *.png files, for 'stack', 'tif', and 'png' inputformats, respectively.

        dims: tuple of ints, optional
            Gives expected shape of a single file of input stack data (for example, x,y,z dimensions for 3d image
            files.) Expected to be in numpy 'F' (Fortran/Matlab; column-major) convention. Used only for 'stack'
            inputformat.

        inputformat: string, optional, default = 'stack'
            Format of data to be read. Must be either 'stack', 'tif', or 'png'.
        """
        checkparams(inputformat, ['stack', 'png', 'tif', 'tif-stack'])
        loader = ImagesLoader(self._sc)

        if inputformat.lower() == 'stack':
            data = loader.fromStack(datafile, dims, startidx=startidx, stopidx=stopidx)
        elif inputformat.lower() == 'tif':
            data = loader.fromTif(datafile, startidx=startidx, stopidx=stopidx)
        elif inputformat.lower() == 'tif-stack':
            data = loader.fromMultipageTif(datafile, startidx=startidx, stopidx=stopidx)
        else:
            data = loader.fromPng(datafile)

        return data

    def loadImagesStackAsSeries(self, datapath, dims, blockSize="150M", startidx=None, stopidx=None, shuffle=False):
        """
        Load Images data as Series data.
        """
        if shuffle:
            loader = ImagesLoader(self._sc)
            return loader.fromStack(datapath, dims, startidx=startidx, stopidx=stopidx).toSeries(blockSize=blockSize)
        else:
            loader = SeriesLoader(self._sc)
            return loader.fromStack(datapath, dims, blockSize=blockSize, startidx=startidx, stopidx=stopidx)

    def convertImagesStackToSeries(self, datapath, outputdirpath, dims, blockSize="150M", startidx=None, stopidx=None,
                                   shuffle=False, overwrite=False):
        """
        Convert images data to Series data in flat binary format.
        """
        if shuffle:
            loader = ImagesLoader(self._sc)
            loader.fromStack(datapath, dims, startidx=startidx, stopidx=stopidx)\
                .saveAsBinarySeries(outputdirpath, blockSize=blockSize, overwrite=overwrite)
        else:
            loader = SeriesLoader(self._sc)
            loader.saveFromStack(datapath, outputdirpath, dims, blockSize=blockSize, overwrite=overwrite,
                                 startidx=startidx, stopidx=stopidx)

    def makeExample(self, dataset, **opts):
        """
        Make an example data set for testing analyses.

        Options include 'pca', 'kmeans', and 'ica'.
        See thunder.utils.datasets for detailed options.

        Parameters
        ----------
        dataset : str
            Which dataset to generate

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset

        """
        checkparams(dataset, ['kmeans', 'pca', 'ica'])

        return DataSets.make(self._sc, dataset, **opts)

    def loadExample(self, dataset):
        """
        Load a local example data set for testing analyses.

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
            return self.loadSeries(os.path.join(path, 'data/iris/iris.bin'))
        elif dataset == "fish-series":
            return self.loadSeries(os.path.join(path, 'data/fish/bin/'))
        elif dataset == "fish-images":
            return self.loadImages(os.path.join(path, 'data/fish/tif-stack'), inputformat="tif-stack")
        else:
            raise NotImplementedError("Dataset '%s' not found" % dataset)

    def loadExampleEC2(self, dataset):
        """
        Load an example data set from EC2.

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset

        params : Tuple or numpy array
            Parameters or metadata for dataset
        """

        if 'ec' not in self._sc.master:
            raise Exception("must be running on EC2 to load this example data sets")
        elif dataset == "zebrafish-optomotor-response":
            path = 'zebrafish.datasets/optomotor-response/1/'
            data = self.loadSeries("s3n://" + path + 'data/dat_plane*.txt', inputformat='text', minPartitions=1000)
            paramfile = self._sc.textFile("s3n://" + path + "params.json")
            params = json.loads(paramfile.first())
            modelfile = asarray(params['trials'])
            return data, modelfile
        else:
            raise NotImplementedError("dataset '%s' not availiable" % dataset)

    def loadSeriesLocal(self, datafile, inputformat='npy', minPartitions=None, keyfile=None, varname=None):
        """
        Load a Series object from a local file (either npy or MAT format).

        File should contain a 1d or 2d matrix, where each row
        of the input matrix is a record.

        Keys can be provided in a separate file (with variable name 'keys', for MAT files).
        If not provided, linear indices will be used for keys.

        Parameters
        ----------
        datafile : str
            File to import

        varname : str, optional, default = None
            Variable name to load (for MAT files only)

        keyfile : str, optional, default = None
            File containing the keys for each record as another 1d or 2d array

        minPartitions : Int, optional, default = 1
            Number of partitions for RDD
        """

        checkparams(inputformat, ['mat', 'npy'])
        loader = SeriesLoader(self._sc, minPartitions=minPartitions)

        if inputformat.lower() == 'mat':
            if varname is None:
                raise Exception('Must provide variable name for loading MAT files')
            data = loader.fromMatLocal(datafile, varname, keyfile)
        else:
            data = loader.fromNpyLocal(datafile, keyfile)

        return data
""" Simple wrapper for a Spark Context to provide loading functionality """

import os
import json
from numpy import asarray, floor, ceil, shape, arange
from scipy.io import loadmat
from pyspark import SparkContext
from thunder.utils.load import PreProcessor, indtosub
from thunder.utils import DataSets
from thunder.rdds import SeriesLoader, ImagesLoader


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

    def loadSeries(self, datafile, nkeys=3, nvalues=None, inputformat='binary', minPartitions=None):
        """Load a Series RDD from data"""

        params = SeriesLoader.loadConf(datafile)
        if params is None:
            if inputformat == 'binary' and nvalues is None:
                raise Exception('Must specify nvalues if not providing a configuration file')
            loader = SeriesLoader(nkeys=nkeys, nvalues=nvalues, minPartitions=minPartitions)
        else:
            loader = SeriesLoader(nkeys=params['nkeys'], nvalues=params['nvalues'], minPartitions=minPartitions)

        if inputformat == 'text':
            data = loader.fromText(datafile, self._sc)
        elif inputformat == 'binary':
            data = loader.fromBinary(datafile, self._sc)
        else:
            raise Exception('Input format for Series must be binary or text')

        return data

    def loadImages(self, datafile, dims=None, inputformat='stack'):
        """Load an Images RDD from data"""

        loader = ImagesLoader(dims=dims)

        if inputformat == 'stack':
            data = loader.fromStack(datafile, self._sc)
        elif inputformat == 'tif':
            data = loader.fromTif(datafile, self._sc)
        elif inputformat == 'png':
            data = loader.fromPng(datafile, self._sc)
        else:
            raise Exception('Input format for Images must be stack, tif, or png')

        return data

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
            return self.loadSeries(os.path.join(path, 'data/iris.txt'), inputformat="text", minPartitions=1)
        elif dataset == "fish":
            return self.loadSeries(os.path.join(path, 'data/fish.txt'), inputformat="text", minPartitions=1)
        else:
            raise NotImplementedError("Dataset '%s' not found" % dataset)

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

    def loadBinaryLocal(self, datafile, nvalues, nkeys, format, keyfile=None, method=None):
        """
        Load data from a local binary file
        """

        raise NotImplementedError

    def loadArrayLocal(self, values, keys=None, method=None):
        """
        Load data from local arrays
        """

        raise NotImplementedError

    def loadMatLocal(self, datafile, varname, keyfile=None, filter=None, minPartitions=1):
        """
        Load data from a local MAT file, from a variable containing
        either a 1d or 2d matrix, into an RDD of (key,value) pairs.
        Each row of the input matrix will become the value of each record.

        Keys can be provided in an extra MAT file containing a variable 'keys'.
        If not provided, linear indices will be used as keys.

        Parameters
        ----------
        datafile : str
            MAT file to import

        varname : str
            Variable name to load from MAT file

        keyfile : str
            MAT file to import with keys (must contain a variable 'keys')

        filter : str, optional, default = None (no preprocessing)
            Which preprocessing to perform

        minPartitions : Int, optional, default = 1
            Number of partitions for data

        """

        data = loadmat(datafile)[varname]
        if data.ndim > 2:
            raise IOError('input data must be one or two dimensional')
        if keyfile:
            keys = map(lambda x: tuple(x), loadmat(keyfile)['keys'])
        else:
            keys = arange(1, shape(data)[0]+1)

        rdd = self._sc.parallelize(zip(keys, data), minPartitions)

        return preprocess(rdd, method=filter)


def preprocess(data, method=None):

    if method:
        preprocessor = PreProcessor(method)
        return data.mapValues(preprocessor.get)
    else:
        return data
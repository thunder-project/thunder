""" Simple wrapper for a Spark Context to provide loading functionality """

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

    def loadSeries(self, datapath, nkeys=None, nvalues=None, inputformat='binary', minPartitions=None,
                   conffile='conf.json', keytype=None, valuetype=None):
        """
        Loads a Series object from data stored as text or binary files.

        Supports single files or multiple files stored on a local file system, a networked file system (mounted
        and available on all cluster nodes), Amazon S3, or HDFS.

        Parameters
        ----------
        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename. Examples
            of valid datapaths include 'a/local/relative/directory/*.stack", "s3n:///my-s3-bucket/data/mydatafile.tif",
            "/mnt/my/absolute/data/directory/", or "file:///mnt/another/data/directory/".

        nkeys: int, optional (but required if `inputformat` is 'text')
            dimensionality of data keys. (For instance, (x,y,z) keyed data for 3-dimensional image timeseries data.)
            For text data, number of keys must be specified in this parameter; for binary data, number of keys must be
            specified either in this parameter or in a configuration file named by the 'conffile' argument if this
            parameter is not set.

        nvalues: int, optional (but required if `inputformat` is 'text')
            Number of values expected to be read. For binary data, nvalues must be specified either in this parameter
            or in a configuration file named by the 'conffile' argument if this parameter is not set.

        inputformat: {'text', 'binary'}. optional, default 'binary'
            Format of data to be read.

        minPartitions: int, optional
            Explicitly specify minimum number of Spark partitions to be generated from this data. Used only for
            text data. Default is to use minParallelism attribute of Spark context object.

        conffile: string, optional, default 'conf.json'
            Path to JSON file with configuration options including 'nkeys', 'nvalues', 'keytype', and 'valuetype'.
            If a file is not found at the given path, then the base directory given in 'datafile'
            will also be checked. Parameters `nkeys` or `nvalues` that are specified as explicit arguments to this
            method will take priority over those found in conffile if both are present.

        Returns
        -------
        data: thunder.rdds.Series
            A newly-created Series object, wrapping an RDD of series data. This RDD will have as keys an n-tuple
            of int, with n given by `nkeys` or the configuration passed in `conffile`. RDD values will be a numpy
            array of length `nvalues` (or as specified in the passed configuration file).
        """
        checkparams(inputformat, ['text', 'binary'])

        from thunder.rdds.fileio.seriesloader import SeriesLoader
        loader = SeriesLoader(self._sc, minPartitions=minPartitions)

        if inputformat.lower() == 'text':
            data = loader.fromText(datapath, nkeys=nkeys)
        else:
            # must be either 'text' or 'binary'
            data = loader.fromBinary(datapath, conffilename=conffile, nkeys=nkeys, nvalues=nvalues,
                                     keytype=keytype, valuetype=valuetype)

        return data

    def loadImages(self, datapath, dims=None, inputformat='stack', dtype='int16', startidx=None, stopidx=None):
        """
        Loads an Images object from data stored as a binary image stack, tif, tif-stack, or png files.

        Supports single files or multiple files, stored on a local file system, a networked file sytem
        (mounted and available on all nodes), or Amazon S3. HDFS is not currently supported for image file data.

        Parameters
        ----------
        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename. Examples
            of valid datapaths include 'a/local/relative/directory/*.stack", "s3n:///my-s3-bucket/data/mydatafile.tif",
            "/mnt/my/absolute/data/directory/", or "file:///mnt/another/data/directory/".

        dims: tuple of positive int, optional (but required if inputformat is 'stack')
            Dimensions of input image data, similar to a numpy 'shape' parameter, for instance (1024, 1024, 48). Binary
            stack data will be interpreted as coming from a multidimensional array of the specified dimensions. Stack
            data should be stored in row-major order (Fortran or Matlab convention) rather than column-major order (C
            or python/numpy convention), where the first dimension corresponds to that which is changing most rapidly
            on disk. So for instance given dims of (x, y, z), the coordinates of the data in a binary stack file
            should be ordered as [(x0, y0, z0), (x1, y0, zo), ..., (xN, y0, z0), (x0, y1, z0), (x1, y1, z0), ...,
            (xN, yM, z0), (x0, y0, z1), ..., (xN, yM, zP)].
            If inputformat is 'png', 'tif', or'tif-stack', the dims parameter (if any) will be ignored; data dimensions
            will instead be read out from the image file headers.

        inputformat: {'stack', 'png', 'tif', 'tif-stack'}. optional, default 'stack'
            Expected format of the input data. 'stack' indicates flat files of raw binary data. 'png' or 'tif' indicate
            two-dimensional image files of the corresponding formats. 'tif-stack' indicates a sequence of multipage tif
            files, with each page of the tif corresponding to a separate z-plane.
            For all formats, separate files are interpreted as distinct time points, with ordering given by
            lexicographic sorting of file names.
            This method assumes that stack data consists of signed 16-bit integers in native byte order. Data types of
            image file data will be as specified in the file headers.

        dtype: string or numpy dtype. optional, default 'int16'
            Data type of the image files to be loaded, specified as a numpy "dtype" string. If inputformat is
            'tif-stack', the dtype parameter (if any) will be ignored; data type will instead be read out from the
            tif headers.

        startidx: nonnegative int, optional
            startidx and stopidx are convenience parameters to allow only a subset of input files to be read in. These
            parameters give the starting index (inclusive) and final index (exclusive) of the data files to be used
            after lexicographically sorting all input data files matching the datapath argument. For example,
            startidx=None (the default) and stopidx=10 will cause only the first 10 data files in datapath to be read
            in; startidx=2 and stopidx=3 will cause only the third file (zero-based index of 2) to be read in. startidx
            and stopidx use the python slice indexing convention (zero-based indexing with an exclusive final position).

        stopidx: nonnegative int, optional
            See startidx.

        Returns
        -------
        data: thunder.rdds.Images
            A newly-created Images object, wrapping an RDD of <int index, numpy array> key-value pairs.

        """
        checkparams(inputformat, ['stack', 'png', 'tif', 'tif-stack'])

        from thunder.rdds.fileio.imagesloader import ImagesLoader
        loader = ImagesLoader(self._sc)

        if inputformat.lower() == 'stack':
            data = loader.fromStack(datapath, dims, dtype=dtype, startidx=startidx, stopidx=stopidx)
        elif inputformat.lower() == 'tif':
            data = loader.fromTif(datapath, startidx=startidx, stopidx=stopidx)
        elif inputformat.lower() == 'tif-stack':
            data = loader.fromMultipageTif(datapath, startidx=startidx, stopidx=stopidx)
        else:
            data = loader.fromPng(datapath)

        return data

    def loadImagesAsSeries(self, datapath, dims=None, inputformat='stack', dtype='int16',
                           blockSize="150M", blockSizeUnits="pixels", startidx=None, stopidx=None, shuffle=False):
        """
        Load Images data as Series data.

        Parameters
        ----------
        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename. Examples
            of valid datapaths include 'a/local/relative/directory/*.stack", "s3n:///my-s3-bucket/data/mydatafile.tif",
            "/mnt/my/absolute/data/directory/", or "file:///mnt/another/data/directory/".

        dims: tuple of positive int, optional (but required if inputformat is 'stack')
            Dimensions of input image data, for instance (1024, 1024, 48). Binary stack data will be interpreted as
            coming from a multidimensional array of the specified dimensions.

            The first dimension of the passed dims tuple should be the one that is changing most rapidly
            on disk. So for instance given dims of (x, y, z), the coordinates of the data in a binary stack file
            should be ordered as [(x0, y0, z0), (x1, y0, z0), ..., (xN, y0, z0), (x0, y1, z0), (x1, y1, z0), ...,
            (xN, yM, z0), (x0, y0, z1), ..., (xN, yM, zP)]. This is the opposite convention from that used by numpy,
            which by default has the fastest-changing dimension listed last (column-major convention). Thus, if loading
            a numpy array `ary`, where `ary.shape == (z, y, x)`, written to disk by `ary.tofile("myarray.stack")`, the
            corresponding dims parameter should be (x, y, z).
            If inputformat is 'tif-stack', the dims parameter (if any) will be ignored; data dimensions will instead
            be read out from the tif file headers.

        inputformat: {'stack', 'tif-stack'}. optional, default 'stack'
            Expected format of the input data. 'stack' indicates flat files of raw binary data, while 'tif-stack'
            indicates a sequence of multipage tif files, with each page of the tif corresponding to a separate z-plane.
            For both stacks and tif stacks, separate files are interpreted as distinct time points, with ordering
            given by lexicographic sorting of file names.
            This method assumes that stack data consists of signed 16-bit integers in native byte order.

        dtype: string or numpy dtype. optional, default 'int16'
            Data type of the image files to be loaded, specified as a numpy "dtype" string. If inputformat is
            'tif-stack', the dtype parameter (if any) will be ignored; data type will instead be read out from the
            tif headers.

        blocksize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of individual output files in bytes (or kilobytes, megabytes, gigabytes). If shuffle=True,
            blocksize can also be a tuple of int specifying either the number of pixels or of splits per dimension to
            apply to the loaded images, or an instance of BlockingStrategy. Whether a tuple of int is interpreted as
            pixels or as splits depends on the value of the blockSizeUnits parameter. blocksize also indirectly
            controls the number of Spark partitions to be used, with one partition used per block created.

        blockSizeUnits: string, either "pixels" or "splits" (or unique prefix of each, such as "s"), default "pixels"
            Specifies units to be used in interpreting a tuple passed as blockSizeSpec when shuffle=True. If a string
            or a BlockingStrategy instance is passed as blockSizeSpec, or if shuffle=False, this parameter has no
            effect.

        startidx: nonnegative int, optional
            startidx and stopidx are convenience parameters to allow only a subset of input files to be read in. These
            parameters give the starting index (inclusive) and final index (exclusive) of the data files to be used
            after lexicographically sorting all input data files matching the datapath argument. For example,
            startidx=None (the default) and stopidx=10 will cause only the first 10 data files in datapath to be read
            in; startidx=2 and stopidx=3 will cause only the third file (zero-based index of 2) to be read in. startidx
            and stopidx use the python slice indexing convention (zero-based indexing with an exclusive final position).

        stopidx: nonnegative int, optional
            See startidx.

        shuffle: boolean, optional, default False
            Controls whether the conversion from Images to Series formats will make use of a Spark shuffle-based method.
            The default at present is not to use a shuffle. The shuffle-based method may lead to higher performance in
            some cases, but the default method appears to be more stable with larger data set sizes. This default may
            change in future releases.

        Returns
        -------
        data: thunder.rdds.Series
            A newly-created Series object, wrapping an RDD of timeseries data generated from the images in datapath.
            This RDD will have as keys an n-tuple of int, with n given by the dimensionality of the original images. The
            keys will be the zero-based spatial index of the timeseries data in the RDD value. The value will be
            a numpy array of length equal to the number of image files loaded. Each loaded image file will contribute
            one point to this value array, with ordering as implied by the lexicographic ordering of image file names.
        """
        checkparams(inputformat, ['stack', 'tif-stack'])

        if inputformat.lower() == 'stack' and not dims:
            raise ValueError("Dimensions ('dims' parameter) must be specified if loading from binary image stack" +
                             " ('stack' value for 'inputformat' parameter)")

        if shuffle:
            from thunder.rdds.fileio.imagesloader import ImagesLoader
            loader = ImagesLoader(self._sc)
            if inputformat.lower() == 'stack':
                images = loader.fromStack(datapath, dims, dtype=dtype, startidx=startidx, stopidx=stopidx)
            else:
                # tif stack
                images = loader.fromMultipageTif(datapath, startidx=startidx, stopidx=stopidx)
            return images.toBlocks(blockSize, units=blockSizeUnits).toSeries()

        else:
            from thunder.rdds.fileio.seriesloader import SeriesLoader
            loader = SeriesLoader(self._sc)
            if inputformat.lower() == 'stack':
                return loader.fromStack(datapath, dims, datatype=dtype, blockSize=blockSize,
                                        startidx=startidx, stopidx=stopidx)
            else:
                # tif stack
                return loader.fromMultipageTif(datapath, blockSize=blockSize,
                                               startidx=startidx, stopidx=stopidx)

    def convertImagesToSeries(self, datapath, outputdirpath, dims=None, inputformat='stack',
                              dtype='int16', blocksize="150M", blockSizeUnits="pixels", startidx=None, stopidx=None,
                              shuffle=False, overwrite=False):
        """
        Write out Images data as Series data, saved in a flat binary format.

        The resulting Series data files may subsequently be read in using the loadSeries() method. The Series data
        object that results will be equivalent to that which would be generated by loadImagesAsSeries(). It is expected
        that loading Series data directly from the series flat binary format, using loadSeries(), will be faster than
        converting image data to a Series object through loadImagesAsSeries().

        Parameters
        ----------
        datapath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A datapath argument may include a single '*' wildcard character in the filename. Examples
            of valid datapaths include 'a/local/relative/directory/*.stack", "s3n:///my-s3-bucket/data/mydatafile.tif",
            "/mnt/my/absolute/data/directory/", or "file:///mnt/another/data/directory/".

        outputdirpath: string
            Path to a directory into which to write Series file output. An outputdir argument may be either a path
            on the local file system or a URI-like format, as in datapath. Examples of valid outputdirpaths include
            "a/relative/directory/", "s3n:///my-s3-bucket/data/myoutput/", or "file:///mnt/a/new/directory/". If the
            directory specified by outputdirpath already exists and the 'overwrite' parameter is False, this method
            will throw a ValueError. If the directory exists and 'overwrite' is True, the existing directory and all
            its contents will be deleted and overwritten.

        dims: tuple of positive int, optional (but required if inputformat is 'stack')
            Dimensions of input image data, for instance (1024, 1024, 48). Binary stack data will be interpreted as
            coming from a multidimensional array of the specified dimensions.

            The first dimension of the passed dims tuple should be the one that is changing most rapidly
            on disk. So for instance given dims of (x, y, z), the coordinates of the data in a binary stack file
            should be ordered as [(x0, y0, z0), (x1, y0, z0), ..., (xN, y0, z0), (x0, y1, z0), (x1, y1, z0), ...,
            (xN, yM, z0), (x0, y0, z1), ..., (xN, yM, zP)]. This is the opposite convention from that used by numpy,
            which by default has the fastest-changing dimension listed last (column-major convention). Thus, if loading
            a numpy array `ary`, where `ary.shape == (z, y, x)`, written to disk by `ary.tofile("myarray.stack")`, the
            corresponding dims parameter should be (x, y, z).
            If inputformat is 'tif-stack', the dims parameter (if any) will be ignored; data dimensions will instead
            be read out from the tif file headers.

        inputformat: {'stack', 'tif-stack'}. optional, default 'stack'
            Expected format of the input data. 'stack' indicates flat files of raw binary data, while 'tif-stack'
            indicates a sequence of multipage tif files, with each page of the tif corresponding to a separate z-plane.
            For both stacks and tif stacks, separate files are interpreted as distinct time points, with ordering
            given by lexicographic sorting of file names.
            This method assumes that stack data consists of signed 16-bit integers in native byte order. The lower-level
            API method SeriesLoader.saveFromStack() allows alternative data types to be read in.

        dtype: string or numpy dtype. optional, default 'int16'
            Data type of the image files to be loaded, specified as a numpy "dtype" string. If inputformat is
            'tif-stack', the dtype parameter (if any) will be ignored; data type will instead be read out from the
            tif headers.

        blocksize: string formatted as e.g. "64M", "512k", "2G", or positive int, tuple of positive int, or instance of
                   BlockingStrategy. optional, default "150M"
            Requested size of individual output files in bytes (or kilobytes, megabytes, gigabytes). blocksize can also
            be an instance of blockingStrategy, or a tuple of int specifying either the number of pixels or of splits
            per dimension to apply to the loaded images. Whether a tuple of int is interpreted as pixels or as splits
            depends on the value of the blockSizeUnits parameter.  This parameter also indirectly controls the number
            of Spark partitions to be used, with one partition used per block created.

        blockSizeUnits: string, either "pixels" or "splits" (or unique prefix of each, such as "s"), default "pixels"
            Specifies units to be used in interpreting a tuple passed as blockSizeSpec when shuffle=True. If a string
            or a BlockingStrategy instance is passed as blockSizeSpec, or if shuffle=False, this parameter has no
            effect.

        startidx: nonnegative int, optional
            startidx and stopidx are convenience parameters to allow only a subset of input files to be read in. These
            parameters give the starting index (inclusive) and final index (exclusive) of the data files to be used
            after lexicographically sorting all input data files matching the datapath argument. For example,
            startidx=None (the default) and stopidx=10 will cause only the first 10 data files in datapath to be read
            in; startidx=2 and stopidx=3 will cause only the third file (zero-based index of 2) to be read in. startidx
            and stopidx use the python slice indexing convention (zero-based indexing with an exclusive final position).

        stopidx: nonnegative int, optional
            See startidx.

        shuffle: boolean, optional, default False
            Controls whether the conversion from Images to Series formats will make use of a Spark shuffle-based method.
            The default at present is not to use a shuffle. The shuffle-based method may lead to higher performance in
            some cases, but the default method appears to be more stable with larger data set sizes. This default may
            change in future releases.

        overwrite: boolean, optional, default False
            If true, the directory specified by outputdirpath will first be deleted, along with all its contents, if it
            already exists. (Use with caution.) If false, a ValueError will be thrown if outputdirpath is found to
            already exist.
        """
        checkparams(inputformat, ['stack', 'tif-stack'])

        if inputformat.lower() == 'stack' and not dims:
            raise ValueError("Dimensions ('dims' parameter) must be specified if loading from binary image stack" +
                             " ('stack' value for 'inputformat' parameter)")

        if shuffle:
            from thunder.rdds.fileio.imagesloader import ImagesLoader
            loader = ImagesLoader(self._sc)
            if inputformat.lower() == 'stack':
                images = loader.fromStack(datapath, dims, dtype=dtype, startidx=startidx, stopidx=stopidx)
            else:
                images = loader.fromMultipageTif(datapath, startidx=startidx, stopidx=stopidx)

            images.toBlocks(blocksize, units=blockSizeUnits).saveAsBinarySeries(outputdirpath, overwrite=overwrite)
        else:
            from thunder.rdds.fileio.seriesloader import SeriesLoader
            loader = SeriesLoader(self._sc)
            if inputformat.lower() == 'stack':
                loader.saveFromStack(datapath, outputdirpath, dims, datatype=dtype,
                                     blockSize=blocksize, overwrite=overwrite, startidx=startidx, stopidx=stopidx)
            else:
                loader.saveFromMultipageTif(datapath, outputdirpath, blockSize=blocksize,
                                            startidx=startidx, stopidx=stopidx, overwrite=overwrite)

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

        import os

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

        import json

        if 'ec' not in self._sc.master:
            raise Exception("must be running on EC2 to load this example data sets")
        elif dataset == "zebrafish-optomotor-response":
            path = 'zebrafish.datasets/optomotor-response/1/'
            data = self.loadSeries("s3n://" + path + 'data/dat_plane*.txt', inputformat='text', minPartitions=1000, nkeys=3)
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

        from thunder.rdds.fileio.seriesloader import SeriesLoader
        loader = SeriesLoader(self._sc, minPartitions=minPartitions)

        if inputformat.lower() == 'mat':
            if varname is None:
                raise Exception('Must provide variable name for loading MAT files')
            data = loader.fromMatLocal(datafile, varname, keyfile)
        else:
            data = loader.fromNpyLocal(datafile, keyfile)

        return data

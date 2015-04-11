""" Simple wrapper for a Spark Context to provide loading functionality """

import os

from thunder.utils.common import checkParams, handleFormat, raiseErrorIfPathExists
from thunder.utils.datasets import DataSets
from thunder.utils.params import Params


class ThunderContext():
    """
    Wrapper for a SparkContext that provides an entry point for loading and saving.

    Also supports creation of example datasets, and loading example
    data both locally and from EC2.

    Attributes
    ----------
    `_sc` : SparkContext
        Spark context for Spark functionality

    `_credentials` : AWSCredentials object, optional, default = None
        Stores public and private keys for AWS services. Typically available through
        configuration files, and but can optionally be set using :func:`ThunderContext.setAWSCredentials()`.
    """
    def __init__(self, sparkcontext):
        self._sc = sparkcontext
        self._credentials = None

    @classmethod
    def start(cls, *args, **kwargs):
        """
        Starts a ThunderContext using the same arguments as SparkContext
        """
        from pyspark import SparkContext
        return ThunderContext(SparkContext(*args, **kwargs))

    def loadSeries(self, dataPath, nkeys=None, nvalues=None, inputFormat='binary', minPartitions=None,
                   confFilename='conf.json', keyType=None, valueType=None, keyPath=None, varName=None):
        """
        Loads a Series object from data stored as binary, text, npy, or mat.

        For binary and text, supports single files or multiple files stored on a local file system,
        a networked file system (mounted and available on all cluster nodes), Amazon S3, or HDFS.
        For local formats (npy and mat) only local file systems currently supported.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, as either a local filesystem path or a URI.
            May include a single '*' wildcard in the filename. Examples of valid dataPaths include
            'local/directory/*.stack", "s3n:///my-s3-bucket/data/", or "file:///mnt/another/directory/".

        nkeys: int, optional (required if `inputFormat` is 'text'), default = None
            Number of keys per record (e.g. 3 for (x, y, z) coordinate keys). Must be specified for
            text data; can be specified here or in a configuration file for binary data.

        nvalues: int, optional (required if `inputFormat` is 'text')
            Number of values per record. Must be specified here or in a configuration file for binary data.

        inputFormat: {'text', 'binary', 'npy', 'mat'}. optional, default = 'binary'
            inputFormat of data to be read.

        minPartitions: int, optional, default = SparkContext.minParallelism
            Minimum number of Spark partitions to use, only for text.

        confFilename: string, optional, default 'conf.json'
            Path to JSON file with configuration options including 'nkeys', 'nvalues',
            'keyType', and 'valueType'. If a file is not found at the given path, then the base
            directory in 'dataPath' will be checked. Parameters will override the conf file.

        keyType: string or numpy dtype, optional, default = None
            Numerical type of keys, will override conf file.

        valueType: string or numpy dtype, optional, default = None
            Numerical type of values, will override conf file.

        keyPath: string, optional, default = None
            Path to file with keys when loading from npy or mat.

        varName : str, optional, default = None
            Variable name to load (for MAT files only)

        Returns
        -------
        data: thunder.rdds.Series
            A Series object, wrapping an RDD, with (n-tuples of ints) : (numpy array) pairs
        """
        checkParams(inputFormat, ['text', 'binary', 'npy', 'mat'])

        from thunder.rdds.fileio.seriesloader import SeriesLoader
        loader = SeriesLoader(self._sc, minPartitions=minPartitions)

        if inputFormat.lower() == 'binary':
            data = loader.fromBinary(dataPath, confFilename=confFilename, nkeys=nkeys, nvalues=nvalues,
                                     keyType=keyType, valueType=valueType)
        elif inputFormat.lower() == 'text':
            if nkeys is None:
                raise Exception('Must provide number of keys per record for loading from text')
            data = loader.fromText(dataPath, nkeys=nkeys)
        elif inputFormat.lower() == 'npy':
            data = loader.fromNpyLocal(dataPath, keyPath)
        else:
            if varName is None:
                raise Exception('Must provide variable name for loading MAT files')
            data = loader.fromMatLocal(dataPath, varName, keyPath)

        return data

    def loadImages(self, dataPath, dims=None, dtype=None, inputFormat='stack', ext=None,
                   startIdx=None, stopIdx=None, recursive=False, nplanes=None, npartitions=None,
                   renumber=False, confFilename='conf.json'):
        """
        Loads an Images object from data stored as a binary image stack, tif, or png files.

        Supports single files or multiple files, stored on a local file system, a networked file sytem
        (mounted and available on all nodes), or Amazon S3. HDFS is not currently supported for image file data.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, as either a local filesystem path or a URI.
            May include a single '*' wildcard in the filename. Examples of valid dataPaths include
            'local/directory/*.stack", "s3n:///my-s3-bucket/data/", or "file:///mnt/another/directory/".

        dims: tuple of positive int, optional (required if inputFormat is 'stack')
            Image dimensions. Binary stack data will be interpreted as a multidimensional array
            with the given dimensions, and should be stored in row-major order (Fortran or Matlab convention),
            where the first dimension changes most rapidly. For 'png' or 'tif' data dimensions
            will be read from the image file headers.

        inputFormat: str, optional, default = 'stack'
            Expected format of the input data: 'stack', 'png', or 'tif'. 'stack' indicates flat binary stacks.
            'png' or 'tif' indicate image format. Page of a multipage tif file will be extend along
            the third dimension. Separate files interpreted as distinct records, with ordering
            given by lexicographic sorting of file names.

        ext: string, optional, default = None
            File extension, default will be "bin" if inputFormat=="stack", "tif" for inputFormat=='tif',
            and 'png' for inputFormat=="png".

        dtype: string or numpy dtype, optional, default = 'int16'
            Data type of the image files to be loaded, specified as a numpy "dtype" string.
            Ignored for 'tif' or 'png' (data will be inferred from image formats).

        startIdx: nonnegative int, optional, default = None
            Convenience parameters to read only a subset of input files. Uses python slice conventions
            (zero-based indexing with exclusive final position). These parameters give the starting
            and final index after lexicographic sorting.

        stopIdx: nonnegative int, optional, default = None
            See startIdx.

        recursive: boolean, optional, default = False
            If true, will recursively descend directories rooted at dataPath, loading all files
            in the tree with an appropriate extension.

        nplanes: positive integer, optional, default = None
            Subdivide individual image files. Every `nplanes` from each file will be considered a new record.
            With nplanes=None (the default), a single file will be considered as representing a single record.
            If the number of records per file is not the same across all files, then `renumber` should be set
            to True to ensure consistent keys.

        npartitions: positive int, optional, default = None
            Specify number of partitions for the RDD, if unspecified will use 1 partition per image.

        renumber: boolean, optional, default = False
            Recalculate keys for records after images are loading. Only necessary if different files contain
            different number of records (e.g. due to specifying nplanes). See Images.renumber().

        confFilename : string, optional, default = 'conf.json'
            Name of conf file if using to specify parameters for binary stack data

        Returns
        -------
        data: thunder.rdds.Images
            An Images object, wrapping an RDD of with (int) : (numpy array) pairs

        """
        checkParams(inputFormat, ['stack', 'png', 'tif', 'tif-stack'])

        from thunder.rdds.fileio.imagesloader import ImagesLoader
        loader = ImagesLoader(self._sc)

        # Checking StartIdx is smaller or equal to StopIdx
        if startIdx is not None and stopIdx is not None and startIdx > stopIdx:
            raise Exception("Error. startIdx {} is larger than stopIdx {}".inputFormat(startIdx, stopIdx))

        if not ext:
            ext = DEFAULT_EXTENSIONS.get(inputFormat.lower(), None)

        if inputFormat.lower() == 'stack':
            data = loader.fromStack(dataPath, dims=dims, dtype=dtype, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                    recursive=recursive, nplanes=nplanes, npartitions=npartitions,
                                    confFilename=confFilename)
        elif inputFormat.lower().startswith('tif'):
            data = loader.fromTif(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx, recursive=recursive,
                                  nplanes=nplanes, npartitions=npartitions)
        else:
            if nplanes:
                raise NotImplementedError("nplanes argument is not supported for png files")
            data = loader.fromPng(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx, recursive=recursive,
                                  npartitions=npartitions)

        if not renumber:
            return data
        else:
            return data.renumber()

    def loadImagesOCP(self, bucketName, resolution, server='ocp.me', startIdx=None, stopIdx=None,
                      minBound=None, maxBound=None):
        """
        Load Images from OCP (Open Connectome Project).

        The OCP is a web service for access to EM brain images and other neural image data.
        The web-service can be accessed at http://www.openconnectomeproject.org/.
        
        Parameters
        ----------
        bucketName: string
            Token name for the project in OCP. This name should exist on the server from which data is loaded.

        resolution: nonnegative int
            Resolution of the data in OCP

        server: string, optional, default = 'ocp.me'
            Name of the OCP server with the specified token.
        
        startIdx: nonnegative int, optional, default = None
            Convenience parameters to read only a subset of input files. Uses python slice conventions
            (zero-based indexing with exclusive final position).

        stopIdx: nonnegative int, optional
            See startIdx.
        
        minBound, maxBound: tuple of nonnegative int, optional, default = None
            X,Y,Z bounds of the data to fetch from OCP. minBound contains the (xMin,yMin,zMin) while
            maxBound contains (xMax,yMax,zMax).

        Returns
        -------
        data: thunder.rdds.Images
             An Images object, wrapping an RDD of with (int) : (numpy array) pairs
        """
      
        from thunder.rdds.fileio.imagesloader import ImagesLoader
        loader = ImagesLoader(self._sc)
        
        # Checking StartIdx is smaller or equal to StopIdx
        if startIdx is not None and stopIdx is not None and startIdx > stopIdx:
            raise Exception("Error. startIdx {} is larger than stopIdx {}".format(startIdx, stopIdx))
        data = loader.fromOCP(bucketName, resolution=resolution, server=server, startIdx=startIdx,
                              stopIdx=stopIdx, minBound=minBound, maxBound=maxBound)

        return data

    def loadImagesAsSeries(self, dataPath, dims=None, inputFormat='stack', ext=None, dtype='int16',
                           blockSize="150M", blockSizeUnits="pixels", startIdx=None, stopIdx=None,
                           shuffle=True, recursive=False, nplanes=None, npartitions=None,
                           renumber=False, confFilename='conf.json'):
        """
        Load Images data as Series data.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, as either a local filesystem path or a URI.
            May include a single '*' wildcard in the filename. Examples of valid dataPaths include
            'local/directory/*.stack", "s3n:///my-s3-bucket/data/", or "file:///mnt/another/directory/".

        dims: tuple of positive int, optional (required if inputFormat is 'stack')
            Image dimensions. Binary stack data will be interpreted as a multidimensional array
            with the given dimensions, and should be stored in row-major order (Fortran or Matlab convention),
            where the first dimension changes most rapidly. For 'png' or 'tif' data dimensions
            will be read from the image file headers.

        inputFormat: str, optional, default = 'stack'
            Expected format of the input data: 'stack', 'png', or 'tif'. 'stack' indicates flat binary stacks.
            'png' or 'tif' indicate image formats. Page of a multipage tif file will be extend along
            the third dimension. Separate files interpreted as distinct records, with ordering
            given by lexicographic sorting of file names.

        ext: string, optional, default = None
            File extension, default will be "bin" if inputFormat=="stack", "tif" for inputFormat=='tif',
            and 'png' for inputFormat=="png".

        dtype: string or numpy dtype. optional, default 'int16'
            Data type of the image files to be loaded, specified as a numpy "dtype" string.
            Ignored for 'tif' or 'png' (data will be inferred from image formats).

        blockSize: string or positive int, optional, default "150M"
            Requested size of blocks (e.g "64M", "512k", "2G"). If shuffle=True, can also be a
            tuple of int specifying the number of pixels or splits per dimension. Indirectly
            controls the number of Spark partitions, with one partition per block.

        blockSizeUnits: string, either "pixels" or "splits", default "pixels"
            Units for interpreting a tuple passed as blockSize when shuffle=True.

        startIdx: nonnegative int, optional, default = None
            Convenience parameters to read only a subset of input files. Uses python slice conventions
            (zero-based indexing with exclusive final position). These parameters give the starting
            and final index after lexicographic sorting.

        stopIdx: nonnegative int, optional, default = None
            See startIdx.

        shuffle: boolean, optional, default = True
            Controls whether the conversion from Images to Series formats will use of a Spark shuffle-based method.

        recursive: boolean, optional, default = False
            If true, will recursively descend directories rooted at dataPath, loading all files
            in the tree with an appropriate extension.

        nplanes: positive integer, optional, default = None
            Subdivide individual image files. Every `nplanes` from each file will be considered a new record.
            With nplanes=None (the default), a single file will be considered as representing a single record.
            If the number of records per file is not the same across all files, then `renumber` should be set
            to True to ensure consistent keys.

        npartitions: positive int, optional, default = None
            Specify number of partitions for the RDD, if unspecified will use 1 partition per image.

        renumber: boolean, optional, default = False
            Recalculate keys for records after images are loading. Only necessary if different files contain
            different number of records (e.g. due to specifying nplanes). See Images.renumber().

        confFilename : string, optional, default = 'conf.json'
            Name of conf file if using to specify parameters for binary stack data

        Returns
        -------
        data: thunder.rdds.Series
            A Series object, wrapping an RDD, with (n-tuples of ints) : (numpy array) pairs.
            Keys will be n-tuples of int, with n given by dimensionality of the images, and correspond
            to indexes into the image arrays. Value will have length equal to the number of image files.
            With each image contributing one point to this value array, with ordering given by
            the lexicographic ordering of image file names.
        """
        checkParams(inputFormat, ['stack', 'tif', 'tif-stack'])

        if not ext:
            ext = DEFAULT_EXTENSIONS.get(inputFormat.lower(), None)

        if shuffle:
            from thunder.rdds.fileio.imagesloader import ImagesLoader
            loader = ImagesLoader(self._sc)
            if inputFormat.lower() == 'stack':
                images = loader.fromStack(dataPath, dims, dtype=dtype, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                          recursive=recursive, nplanes=nplanes, npartitions=npartitions,
                                          confFilename=confFilename)
            else:
                # tif / tif stack
                images = loader.fromTif(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                        recursive=recursive, nplanes=nplanes, npartitions=npartitions)
            if renumber:
                images = images.renumber()
            return images.toBlocks(blockSize, units=blockSizeUnits).toSeries()

        else:
            from thunder.rdds.fileio.seriesloader import SeriesLoader
            if nplanes is not None:
                raise NotImplementedError("nplanes is not supported with shuffle=False")
            if npartitions is not None:
                raise NotImplementedError("npartitions is not supported with shuffle=False")
            if renumber:
                raise NotImplementedError("renumber is not supported with shuffle=False")

            loader = SeriesLoader(self._sc)
            if inputFormat.lower() == 'stack':
                return loader.fromStack(dataPath, dims, ext=ext, dtype=dtype, blockSize=blockSize,
                                        startIdx=startIdx, stopIdx=stopIdx, recursive=recursive)
            else:
                # tif / tif stack
                return loader.fromTif(dataPath, ext=ext, blockSize=blockSize,
                                      startIdx=startIdx, stopIdx=stopIdx, recursive=recursive)

    def convertImagesToSeries(self, dataPath, outputDirPath, dims=None, inputFormat='stack', ext=None,
                              dtype='int16', blockSize="150M", blockSizeUnits="pixels", startIdx=None, stopIdx=None,
                              shuffle=True, overwrite=False, recursive=False, nplanes=None, npartitions=None,
                              renumber=False, confFilename='conf.json'):
        """
        Write out Images data as Series data, saved in a flat binary format.

        The resulting files may subsequently be read in using ThunderContext.loadSeries().
        Loading Series data directly will likely be faster than converting image data
        to a Series object through loadImagesAsSeries().

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, as either a local filesystem path or a URI.
            May include a single '*' wildcard in the filename. Examples of valid dataPaths include
            'local/directory/*.stack", "s3n:///my-s3-bucket/data/", or "file:///mnt/another/directory/".

        outputDirPath: string
            Path to directory to write Series file output. May be either a path on the local file system
            or a URI-like format, such as "local/directory", "s3n:///my-s3-bucket/data/",
            or "file:///mnt/another/directory/". If the directory exists and 'overwrite' is True,
            the existing directory and all its contents will be deleted and overwritten.

        dims: tuple of positive int, optional (required if inputFormat is 'stack')
            Image dimensions. Binary stack data will be interpreted as a multidimensional array
            with the given dimensions, and should be stored in row-major order (Fortran or Matlab convention),
            where the first dimension changes most rapidly. For 'png' or 'tif' data dimensions
            will be read from the image file headers.

        inputFormat: str, optional, default = 'stack'
            Expected format of the input data: 'stack', 'png', or 'tif'. 'stack' indicates flat binary stacks.
            'png' or 'tif' indicate image formats. Page of a multipage tif file will be extend along
            the third dimension. Separate files interpreted as distinct records, with ordering
            given by lexicographic sorting of file names.

        ext: string, optional, default = None
            File extension, default will be "bin" if inputFormat=="stack", "tif" for inputFormat=='tif',
            and 'png' for inputFormat=="png".

        dtype: string or numpy dtype. optional, default 'int16'
            Data type of the image files to be loaded, specified as a numpy "dtype" string.
            Ignored for 'tif' or 'png' (data will be inferred from image formats).

        blockSize: string or positive int, optional, default "150M"
            Requested size of blocks (e.g "64M", "512k", "2G"). If shuffle=True, can also be a
            tuple of int specifying the number of pixels or splits per dimension. Indirectly
            controls the number of Spark partitions, with one partition per block.

        blockSizeUnits: string, either "pixels" or "splits", default "pixels"
            Units for interpreting a tuple passed as blockSize when shuffle=True.

        startIdx: nonnegative int, optional, default = None
            Convenience parameters to read only a subset of input files. Uses python slice conventions
            (zero-based indexing with exclusive final position). These parameters give the starting
            and final index after lexicographic sorting.

        stopIdx: nonnegative int, optional, default = None
            See startIdx.

        shuffle: boolean, optional, default = True
            Controls whether the conversion from Images to Series formats will use of a Spark shuffle-based method.

        overwrite: boolean, optional, default False
            If true, the directory specified by outputDirPath will be deleted (recursively) if it
            already exists. (Use with caution.)

        recursive: boolean, optional, default = False
            If true, will recursively descend directories rooted at dataPath, loading all files
            in the tree with an appropriate extension.

        nplanes: positive integer, optional, default = None
            Subdivide individual image files. Every `nplanes` from each file will be considered a new record.
            With nplanes=None (the default), a single file will be considered as representing a single record.
            If the number of records per file is not the same across all files, then `renumber` should be set
            to True to ensure consistent keys.

        npartitions: positive int, optional, default = None
            Specify number of partitions for the RDD, if unspecified will use 1 partition per image.

        renumber: boolean, optional, default = False
            Recalculate keys for records after images are loading. Only necessary if different files contain
            different number of records (e.g. due to specifying nplanes). See Images.renumber().

        confFilename : string, optional, default = 'conf.json'
            Name of conf file if using to specify parameters for binary stack data

        """
        checkParams(inputFormat, ['stack', 'tif', 'tif-stack'])

        if not overwrite:
            raiseErrorIfPathExists(outputDirPath, awsCredentialsOverride=self._credentials)
            overwrite = True  # prevent additional downstream checks for this path

        if not ext:
            ext = DEFAULT_EXTENSIONS.get(inputFormat.lower(), None)

        if shuffle:
            from thunder.rdds.fileio.imagesloader import ImagesLoader
            loader = ImagesLoader(self._sc)
            if inputFormat.lower() == 'stack':
                images = loader.fromStack(dataPath, dims, ext=ext, dtype=dtype, startIdx=startIdx, stopIdx=stopIdx,
                                          recursive=recursive, nplanes=nplanes, npartitions=npartitions,
                                          confFilename=confFilename)
            else:
                # 'tif' or 'tif-stack'
                images = loader.fromTif(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                        recursive=recursive, nplanes=nplanes, npartitions=npartitions)
            if renumber:
                images = images.renumber()
            images.toBlocks(blockSize, units=blockSizeUnits).saveAsBinarySeries(outputDirPath, overwrite=overwrite)
        else:
            from thunder.rdds.fileio.seriesloader import SeriesLoader
            if nplanes is not None:
                raise NotImplementedError("nplanes is not supported with shuffle=False")
            if npartitions is not None:
                raise NotImplementedError("npartitions is not supported with shuffle=False")
            loader = SeriesLoader(self._sc)
            if inputFormat.lower() == 'stack':
                loader.saveFromStack(dataPath, outputDirPath, dims, ext=ext, dtype=dtype,
                                     blockSize=blockSize, overwrite=overwrite, startIdx=startIdx,
                                     stopIdx=stopIdx, recursive=recursive)
            else:
                # 'tif' or 'tif-stack'
                loader.saveFromTif(dataPath, outputDirPath, ext=ext, blockSize=blockSize,
                                   startIdx=startIdx, stopIdx=stopIdx, overwrite=overwrite,
                                   recursive=recursive)

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
        from thunder.utils.datasets import DATASET_MAKERS
        checkParams(dataset, DATASET_MAKERS.keys())

        return DataSets.make(self._sc, dataset, **opts)

    def loadExample(self, dataset=None):
        """
        Load a local example data set for testing analyses.

        Some of these data sets are extremely downsampled and should be considered
        useful only for testing the API. If called with None,
        will return list of available datasets.

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : Data object
            Generated dataset as a Thunder data objects (e.g Series or Images)
        """
        import atexit
        import shutil
        import tempfile
        from pkg_resources import resource_listdir, resource_filename

        DATASETS = {
            'iris': 'iris',
            'fish-series': 'fish/series',
            'fish-images': 'fish/images',
            'mouse-series': 'mouse/series',
            'mouse-images': 'mouse/images',
            'mouse-params': 'mouse/params'
        }

        if dataset is None:
            return sorted(DATASETS.keys())

        checkParams(dataset, DATASETS.keys())

        if 'ec2' in self._sc.master:
            tmpdir = os.path.join('/root/thunder/python/thunder/utils', 'data', DATASETS[dataset])
        else:
            tmpdir = tempfile.mkdtemp()
            atexit.register(shutil.rmtree, tmpdir)

            def copyLocal(target):
                files = resource_listdir('thunder.utils.data', target)
                for f in files:
                    path = resource_filename('thunder.utils.data', os.path.join(target, f))
                    shutil.copy(path, tmpdir)

            copyLocal(DATASETS[dataset])

        npartitions = self._sc.defaultParallelism

        if dataset == "iris":
            return self.loadSeries(tmpdir)
        elif dataset == "fish-series":
            return self.loadSeries(tmpdir).astype('float')
        elif dataset == "fish-images":
            return self.loadImages(tmpdir, inputFormat="tif", npartitions=npartitions)
        elif dataset == "mouse-series":
            return self.loadSeries(tmpdir).astype('float')
        elif dataset == "mouse-images":
            return self.loadImages(tmpdir, npartitions=npartitions)
        elif dataset == "mouse-params":
            return self.loadParams(os.path.join(tmpdir, 'covariates.json'))

    def loadExampleS3(self, dataset=None):
        """
        Load an example data set from S3.

        Info on the included datasets can be found at the CodeNeuro data repository
        (http://datasets.codeneuro.org/). If called with None, will return
        list of available datasets.

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : a Data object (usually a Series or Images)
            The dataset as one of Thunder's data objects

        params : dict
            Parameters or metadata for dataset
        """
        DATASETS = {
            'ahrens.lab/direction.selectivity': 'ahrens.lab/direction.selectivity/1/',
            'ahrens.lab/optomotor.response': 'ahrens.lab/optomotor.response/1/',
            'svoboda.lab/tactile.navigation': 'svoboda.lab/tactile.navigation/1/'
        }

        if dataset is None:
            return DATASETS.keys()

        if 'local' in self._sc.master:
            raise Exception("Must be running on an EC2 cluster to load this example data set")

        checkParams(dataset, DATASETS.keys())

        basePath = 's3n://neuro.datasets/'
        dataPath = DATASETS[dataset]

        data = self.loadSeries(basePath + dataPath + 'series')
        params = self.loadParams(basePath + dataPath + 'params/covariates.json')

        return data, params

    def loadJSON(self, path):
        """
        Generic function for loading JSON from a path, handling local file systems and S3

        Parameters
        ----------
        path : str
            Path to a file, can be on a local file system or an S3 bucket

        Returns
        -------
        A string with the JSON
        """

        import json
        from thunder.rdds.fileio.readers import getFileReaderForPath, FileNotFoundError

        reader = getFileReaderForPath(path)(awsCredentialsOverride=self._credentials)
        try:
            buffer = reader.read(path)
        except FileNotFoundError:
            raise Exception("Cannot find file %s" % path)

        return json.loads(buffer)

    def loadParams(self, path):
        """
        Load a file with parameters from a local file system or S3.

        Assumes file is JSON with basic types (strings, integers, doubles, lists),
        in either a single dict or list of dict-likes, and each dict has at least
        a "name" field and a "value" field.

        Useful for loading generic meta data, parameters, covariates, etc.

        Parameters
        ----------
        path : str
            Path to file, can be on a local file system or an S3 bucket

        Returns
        -------
        A dict or list with the parameters
        """
        blob = self.loadJSON(path)
        return Params(blob)

    def loadSources(self, path):
        """
        Load a file with sources from a local file system or S3.

        Parameters
        ----------
        path : str
            Path to file, can be on a local file system or an S3 bucket

        Returns
        -------
        A SourceModel

        See also
        --------
        thunder.SourceExtraction
        """
        from thunder import SourceExtraction

        blob = self.loadJSON(path)
        return SourceExtraction.fromJSON(blob)

    def export(self, data, filename, outputFormat=None, overwrite=False, varname=None):
        """
        Export local array data to a variety of formats.

        Can write to a local file sytem or S3 (destination inferred from filename schema).
        S3 writing useful for persisting arrays when working in an environment without
        accessible local storage.

        Parameters
        ----------
        data : array-like
            The data to export

        filename : str
            Output location (path/to/file.ext)

        outputFormat : str, optional, default = None
            Ouput format ("npy", "mat", or "txt"), if not provided will
            try to infer from file extension.

        overwrite : boolean, optional, default = False
            Whether to overwrite if directory or file already exists

        varname : str, optional, default = None
            Variable name for writing "mat" formatted files
        """
        from numpy import save, savetxt, asarray
        from scipy.io import savemat
        from StringIO import StringIO

        from thunder.rdds.fileio.writers import getFileWriterForPath

        path, file, outputFormat = handleFormat(filename, outputFormat)
        checkParams(outputFormat, ["npy", "mat", "txt"])
        clazz = getFileWriterForPath(filename)
        writer = clazz(path, file, overwrite=overwrite, awsCredentialsOverride=self._credentials)

        stream = StringIO()

        if outputFormat == "mat":
            varname = os.path.splitext(file)[0] if varname is None else varname
            savemat(stream, mdict={varname: data}, oned_as='column', do_compression='true')
        if outputFormat == "npy":
            save(stream, data)
        if outputFormat == "txt":
            if asarray(data).ndim > 2:
                raise Exception("Cannot write data with more than two dimensions to text")
            savetxt(stream, data)

        stream.seek(0)
        writer.writeFile(stream.buf)

    def setAWSCredentials(self, awsAccessKeyId, awsSecretAccessKey):
        """
        Manually set AWS access credentials to be used by Thunder.

        Provided for hosted cloud environments without filesystem access.
        If launching a cluster using the thunder-ec2 script, credentials will be configured
        automatically (inside core-site.xml and ~/.boto), so this method should not need to be called.

        Parameters
        ----------
        awsAccessKeyId : string
             AWS public key, usually starts with "AKIA"

        awsSecretAccessKey : string
            AWS private key
        """
        from thunder.utils.common import AWSCredentials
        self._credentials = AWSCredentials(awsAccessKeyId, awsSecretAccessKey)
        self._credentials.setOnContext(self._sc)


DEFAULT_EXTENSIONS = {
    "stack": "bin",
    "tif": "tif",
    "tif-stack": "tif",
    "png": "png",
    "mat": "mat",
    "npy": "npy",
    "txt": "txt"
}

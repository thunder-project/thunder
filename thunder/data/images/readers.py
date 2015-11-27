from io import BytesIO
from numpy import array, dstack, frombuffer, prod, load, swapaxes, random, asarray

from thunder import engine, mode, credentials


def fromrdd(rdd, **kwargs):
    from .images import Images
    return Images(rdd, **kwargs)

def fromlist(items, accessor=None, keys=None, npartitions=None, **kwargs):

    if mode() == 'spark':
        nrecords = len(items)
        if keys:
            items = zip(keys, items)
        else:
            items = enumerate(items)
        if not npartitions:
            npartitions = engine().defaultParallelism
        rdd = engine().parallelize(items, npartitions)
        if accessor:
            rdd = rdd.mapValues(accessor)
        return fromrdd(rdd, nrecords=nrecords, **kwargs)

    else:
        raise NotImplementedError("Loading not implemented for '%s' mode" % mode())

def fromurls(urls, accessor=None, keys=None, npartitions=None, **kwargs):

    if mode() == 'spark':
        if keys:
            urls = zip(keys, urls)
        else:
            urls = enumerate(urls)
        rdd = engine().parallelize(urls, npartitions)
        if accessor:
            rdd = rdd.mapValues(accessor)
        return fromrdd(rdd, **kwargs)

def frompath(path, accessor=None, ext=None, start=None, stop=None, recursive=None,
             npartitions=None, dims=None, dtype=None, recount=False):

    if mode() == 'spark':
        from thunder.data.fileio.readers import getParallelReaderForPath
        reader = getParallelReaderForPath(path)(engine(), credentials=credentials())
        rdd = reader.read(path, ext=ext, start=start, stop=stop,
                          recursive=recursive, npartitions=npartitions)
        if accessor:
            rdd = rdd.flatMap(accessor)
        if recount:
            nrecords = None
        else:
            nrecords = reader.lastNRecs
        return fromrdd(rdd, nrecords=nrecords, dims=dims, dtype=dtype)

    else:
        raise NotImplementedError("Loading not implemented for '%s' mode" % mode())

def fromarray(arrays, npartitions=None):
    """
    Load Images data from passed sequence of numpy arrays.

    Expected usage is mainly in testing - having a full dataset loaded in memory
    on the driver is likely prohibitive in the use cases for which Thunder is intended.
    """

    arrays = asarray(arrays)

    shape = None
    dtype = None
    for ary in arrays:
        if shape is None:
            shape = ary.shape
            dtype = ary.dtype
        if not ary.shape == shape:
            raise ValueError("Arrays must all be of same shape; got both %s and %s" %
                             (str(shape), str(ary.shape)))
        if not ary.dtype == dtype:
            raise ValueError("Arrays must all be of same data type; got both %s and %s" %
                             (str(dtype), str(ary.dtype)))
    narrays = len(arrays)
    npartitions = min(narrays, npartitions) if npartitions else narrays
    return fromlist(arrays, npartitions=npartitions, dims=shape, dtype=str(dtype))


def frombinary(path, dims=None, dtype=None, ext='bin', start=None, stop=None, recursive=False,
               nplanes=None, npartitions=None, conf='conf.json', order='C'):
    """
    Load images from a directory of flat binary files

    The RDD wrapped by the returned Images object will have a number of partitions equal to the number of image data
    files read in by this method.

    Currently all binary data read by this method is assumed to be formatted as signed 16 bit integers in native
    byte order.

    Parameters
    ----------

    path: string
        Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
        including scheme. A dataPath argument may include a single '*' wildcard character in the filename.

    dims: tuple of positive int
        Dimensions of input image data, ordered with fastest-changing dimension first

    ext: string, optional, default "bin"
        Extension required on data files to be loaded.

    start, stop: nonnegative int. optional.
        Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
        `datapath` and `ext`. Interpreted according to python slice indexing conventions.

    recursive: boolean, default False
        If true, will recursively descend directories rooted at datapath, loading all files in the tree that
        have an extension matching 'ext'. Recursive loading is currently only implemented for local filesystems
        (not S3 or GS).

    nplanes: positive integer, default None
        If passed, will cause a single binary stack file to be subdivided into multiple records. Every
        `nplanes` z-planes in the file will be taken as a new record, with the first nplane planes of the
        first file being record 0, the second nplane planes being record 1, etc, until the first file is
        exhausted and record ordering continues with the first nplane planes of the second file, and so on.
        With nplanes=None (the default), a single file will be considered as representing a single record.

    npartitions: positive int, optional.
        If specified, request a certain number of partitions for the underlying Spark RDD. Default is 1
        partition per image file.
    """
    import json
    from thunder.data.fileio.readers import getFileReaderForPath, FileNotFoundError
    try:
        reader = getFileReaderForPath(path)(credentials=credentials())
        buf = reader.read(path, filename=conf)
        params = json.loads(buf)
    except FileNotFoundError:
        params = {}

    if 'dtype' in params.keys():
        dtype = params['dtype']
    if 'dims' in params.keys():
        dims = params['dims']

    if not dims:
        raise ValueError("Image dimensions must be specified either as argument or in a conf.json file")

    if not dtype:
        dtype = 'int16'

    if nplanes is not None:
        if nplanes <= 0:
            raise ValueError("nplanes must be positive if passed, got %d" % nplanes)
        if dims[-1] % nplanes:
            raise ValueError("Last dimension of binary image '%d' must be divisible by nplanes '%d'" %
                             (dims[-1], nplanes))

    def getarray(idxAndBuf):
        idx, buf = idxAndBuf
        ary = frombuffer(buf, dtype=dtype, count=int(prod(dims))).reshape(dims, order=order)
        if nplanes is None:
            yield idx, ary
        else:
            # divide array into chunks of nplanes
            npoints = dims[-1] / nplanes  # integer division
            if dims[-1] % nplanes:
                npoints += 1
            timepoint = 0
            lastPlane = 0
            curPlane = 1
            while curPlane < ary.shape[-1]:
                if curPlane % nplanes == 0:
                    slices = [slice(None)] * (ary.ndim - 1) + [slice(lastPlane, curPlane)]
                    yield idx*npoints + timepoint, ary[slices].squeeze()
                    timepoint += 1
                    lastPlane = curPlane
                curPlane += 1
            # yield remaining planes
            slices = [slice(None)] * (ary.ndim - 1) + [slice(lastPlane, ary.shape[-1])]
            yield idx*npoints + timepoint, ary[slices].squeeze()

    recount = False if nplanes is None else True
    append = [nplanes] if nplanes > 1 else []
    newdims = tuple(list(dims[:-1]) + append) if nplanes else dims
    return frompath(path, accessor=getarray, ext=ext, start=start,
                    stop=stop, recursive=recursive, npartitions=npartitions,
                    dims=newdims, dtype=dtype, recount=recount)

def fromtif(path, ext='tif', start=None, stop=None, recursive=False, nplanes=None, npartitions=None):
    """
    Sets up a new Images object with data to be read from one or more tif files.

    Multiple pages of a multipage tif file will by default be assumed to represent the z-axis (depth) of a
    single 3-dimensional volume, in which case a single input multipage tif file will be converted into
    a single Images record. If `nplanes` is passed, then every nplanes pages will be interpreted as a single
    3d volume (2d if nplanes==1), allowing a single tif file to contain multiple Images records.

    This method attempts to explicitly import PIL. ImportError may be thrown if 'from PIL import Image' is
    unsuccessful. (PIL/pillow is not an explicit requirement for thunder.)

    The RDD wrapped by the returned Images object will by default have a number of partitions equal to the
    number of image data files read in by this method; it may have fewer partitions if npartitions is specified.

    Parameters
    ----------

    path: string
        Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
        including scheme. A datapath argument may include a single '*' wildcard character in the filename.

    ext: string, optional, default "tif"
        Extension required on data files to be loaded.

    start, stop: nonnegative int. optional.
        Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
        `datapath` and `ext`. Interpreted according to python slice indexing conventions.

    recursive: boolean, default False
        If true, will recursively descend directories rooted at datapath, loading all files in the tree that
        have an extension matching 'ext'.

    nplanes: positive integer, default None
        If passed, will cause a single multipage tif file to be subdivided into multiple records. Every
        `nplanes` tif pages in the file will be taken as a new record, with the first nplane pages of the
        first file being record 0, the second nplane pages being record 1, etc, until the first file is
        exhausted and record ordering continues with the first nplane images of the second file, and so on.
        With nplanes=None (the default), a single file will be considered as representing a single record.

    npartitions: positive int, optional.
        If specified, request a certain number of partitions for the underlying Spark RDD. Default is 1
        partition per image file.
    """
    import skimage.external.tifffile as tifffile

    if nplanes is not None and nplanes <= 0:
        raise ValueError("nplanes must be positive if passed, got %d" % nplanes)

    def getarray(idxAndBuf):
        idx, buf = idxAndBuf
        fbuf = BytesIO(buf)
        tfh = tifffile.TiffFile(fbuf)
        ary = tfh.asarray()
        pageCount = ary.shape[0]
        if nplanes is not None:
            values = [ary[i:(i+nplanes)] for i in xrange(0, ary.shape[0], nplanes)]
        else:
            values = [ary]
        tfh.close()

        if ary.ndim == 3:
            values = [val.transpose((1, 2, 0)) for val in values]
            values = [val.squeeze(-1) if val.shape[-1] == 1 else val for val in values]

        if nplanes and (pageCount % nplanes):
            raise ValueError("nplanes '%d' does not evenly divide '%d'" % (nplanes, pageCount))
        nvals = len(values)
        keys = [idx*nvals + timepoint for timepoint in xrange(nvals)]
        return zip(keys, values)

    recount = False if nplanes is None else True
    return frompath(path, accessor=getarray, ext=ext, start=start, stop=stop,
                    recursive=recursive, npartitions=npartitions, recount=recount)

def frompng(path, ext='png', start=None, stop=None, recursive=False, npartitions=None):
    """Load an Images object stored in a directory of png files

    The RDD wrapped by the returned Images object will have a number of partitions equal to the number of image data
    files read in by this method.

    Parameters
    ----------

    dataPath: string
        Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
        including scheme. A dataPath argument may include a single '*' wildcard character in the filename.

    ext: string, optional, default "png"
        Extension required on data files to be loaded.

    start, stop: nonnegative int. optional.
        Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
        `datapath` and `ext`. Interpreted according to python slice indexing conventions.

    recursive: boolean, default False
        If true, will recursively descend directories rooted at datapath, loading all files in the tree that
        have an extension matching 'ext'. Recursive loading is currently only implemented for local filesystems
        (not s3 or Google Storage).

    npartitions: positive int, optional.
        If specified, request a certain number of partitions for the underlying Spark RDD. Default is 1
        partition per image file.
    """
    from scipy.misc import imread

    def getarray(idxAndBuf):
        idx, buf = idxAndBuf
        fbuf = BytesIO(buf)
        yield idx, imread(fbuf)

    return frompath(path, accessor=getarray, ext=ext, start=start,
                    stop=stop, recursive=recursive, npartitions=npartitions)

def fromocp(bucketName, resolution, server='ocp.me', start=None, stop=None,
            minBound=None, maxBound=None):
    """
    Load data from OCP

    Parameters
    ----------
    bucketName: string
        Name of the token/bucket in OCP. You can use the token name you created in OCP here.
        You can also access publicly available data on OCP at this URL "http://ocp.me/ocp/ca/public_tokens/"

    resolution: nonnegative int
        Resolution of the data in OCP

    server: string. optional.
        Name of the server in OCP which has the corresponding token.

    start, stop: nonnegative int. optional.
        Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
        `datapath` and `ext`. Interpreted according to python slice indexing conventions.

    minBound, maxBound: tuple of nonnegative int. optional.
        X,Y,Z bounds of the data you want to fetch from OCP. minBound contains
        the (xMin,yMin,zMin) while maxBound contains (xMax,yMax,zMax)
    """
    # Given a data-path/bucket Query JSON
    # Given bounds get a list of URI's
    import urllib2
    urlList = []
    url = 'http://{}/ocp/ca/{}/info/'.format(server, bucketName)

    try:
        f = urllib2.urlopen(url)
    except urllib2.URLError:
        raise Exception("Failed URL {}".format(url))

    import json
    projInfo = json.loads(f.read())

    # Loading Information from JSON object
    ximageSize, yimageSize = projInfo['dataset']['imagesize']['{}'.format(resolution)]
    zimageStart, zimageStop = projInfo['dataset']['slicerange']
    timageStart, timageStop = projInfo['dataset']['timerange']

    # Checking if dimensions are within bounds
    if start is None:
        start = timageStart
    elif start < timageStart or start > timageStop:
        raise Exception("start out of bounds {},{}".format(timageStart, timageStop))

    if stop is None:
        stop = timageStop
    elif stop < timageStart or stop > timageStop:
        raise Exception("start out of bounds {},{}".format(timageStart, timageStop))

    if minBound is None:
        minBound = (0, 0, zimageStart)
    elif minBound < (0, 0, zimageStart) or minBound > (ximageSize, yimageSize, zimageStop):
        raise Exception("minBound is incorrect {},{}".format((0, 0, zimageStart),
                                                             (ximageSize, yimageSize, zimageStop)))

    if maxBound is None:
        maxBound = (ximageSize, yimageSize, zimageStop)
    elif maxBound < (0, 0, zimageStart) or maxBound > (ximageSize, yimageSize, zimageStop):
        raise Exception("minBound is incorrect {},{}".format((0, 0, zimageStart), (ximageSize, yimageSize,
                                                                                   zimageStop)))

    for t in range(timageStart, timageStop, 1):
        urlList.append("http://{}/ocp/ca/{}/npz/{},{}/{}/{},{}/{},{}/{},{}/".
                       format(server, bucketName, t, t + 1, resolution, minBound[0],
                              maxBound[0], minBound[1], maxBound[1], minBound[2], maxBound[2]))

    def read(target):
        """
        Fetch URL from the server
        """
        try:
            npzFile = urllib2.urlopen(target)
        except urllib2.URLError:
            raise Exception("Failed URL {}.".format(target))

        imgData = npzFile.read()

        import zlib
        import cStringIO
        pageStr = zlib.decompress(imgData[:])
        pageObj = cStringIO.StringIO(pageStr)
        data = load(pageObj)
        # Data comes in as 4d numpy array in t,z,y,x order. Swapping axes and removing the time dimension
        # to give back a 3d numpy array in x,y,z order
        data = swapaxes(data[0, :, :, :], 0, 2)

        return data

    return fromurls(enumerate(urlList), accessor=read, npartitions=len(urlList))

def fromrandom(shape=(10, 50, 50), npartitions=1, seed=42):

    seed = hash(seed)

    def generate(v):
        random.seed(seed + v)
        return random.randn(*shape[1:])

    return fromlist(range(shape[0]), accessor=generate, npartitions=npartitions)

def fromexample(name=None):

    datasets = ['mouse', 'fish']

    if name is None:
        print 'Availiable example datasets'
        for d in datasets:
            print '- ' + d
        return

    if mode() == 'spark':
        print('Downloading data from S3, this may take a few seconds...')

        if name == 'mouse':
            data = frombinary(path='s3n://thunder-sample-data/mouse-images/', npartitions=1, order='F')

        elif name == 'fish':
            data = fromtif(path='s3n://thunder-sample-data/fish-images/', npartitions=1)

        else:
            raise NotImplementedError("Example '%s' not found" % name)

        data.cache()
        data.count()
        return data

    else:
        raise NotImplementedError("Loading not implemented for '%s' mode" % mode())
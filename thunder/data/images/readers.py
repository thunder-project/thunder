import checkist
from io import BytesIO
from numpy import frombuffer, prod, load, swapaxes, random, asarray

from thunder import engine, mode, credentials


def fromrdd(rdd, **kwargs):
    """
    Load images from an Spark RDD.
    """
    from .images import Images
    return Images(rdd, **kwargs)

def fromlist(items, accessor=None, keys=None, npartitions=None, **kwargs):
    """
    Load images from a list of items using the given accessor.

    Parameters
    ----------
    accessor : function
        Apply to each item from the list to yield an image

    keys : list, optional, default=None
        An optional list of keys

    npartitions : int
        Number of partitions for computational engine
    """
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
    """
    Load images from a list of URLs using the given accessor.

    Parameters
    ----------
    accessor : function
        Apply to each item from the list to yield an image

    keys : list, optional, default=None
        An optional list of keys

    npartitions : int
        Number of partitions for computational engine
    """
    if mode() == 'spark':
        if keys:
            urls = zip(keys, urls)
        else:
            urls = enumerate(urls)
        rdd = engine().parallelize(urls, npartitions)
        if accessor:
            rdd = rdd.mapValues(accessor)
        return fromrdd(rdd, **kwargs)

def frompath(path, accessor=None, ext=None, start=None, stop=None, recursive=False,
             npartitions=None, dims=None, dtype=None, recount=False):
    """
    Load images from a path using the given accessor.

    Supports both local and remote filesystems.

    Parameters
    ----------
    accessor : function
        Apply to each item after loading to yield an image.

    ext : str, optional, default=None
        File extension.

    npartitions : int, optional, default=None
        Number of partitions for computational engine,
        if None will use default for engine.

    dims : tuple, optional, default=None
        Dimensions of images.

    dtype : str, optional, default=None
        Numerical type of images.

    start, stop: nonnegative int, optional, default=None
        Indices of files to load, interpreted using Python slicing conventions.

    recursive : boolean, optional, default=False
        If true, will recursively descend directories from path, loading all files
        with an extension matching 'ext'.

    recount : boolean, optional, default=False
        Force subsequent record counting.
    """
    if mode() == 'spark':
        from thunder.data.readers import get_parallel_reader
        reader = get_parallel_reader(path)(engine(), credentials=credentials())
        rdd = reader.read(path, ext=ext, start=start, stop=stop,
                          recursive=recursive, npartitions=npartitions)
        if accessor:
            rdd = rdd.flatMap(accessor)
        if recount:
            nrecords = None
        else:
            nrecords = reader.nfiles
        return fromrdd(rdd, nrecords=nrecords, dims=dims, dtype=dtype)

    else:
        raise NotImplementedError("Loading not implemented for '%s' mode" % mode())

def fromarray(arrays, npartitions=None):
    """
    Load images from a sequence of ndarrays.
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
    Load images from binary files.

    Parameters
    ----------
    path : str
        Path to data files or directory, specified as either a local filesystem path
        or in a URI-like format, including scheme. May include a single '*' wildcard character.

    dims : tuple of positive int
        Dimensions of input image data, ordered with fastest-changing dimension first

    ext : string, optional, default="bin"
        Extension required on data files to be loaded.

    start, stop : nonnegative int, optional, default=None
        Indices of the first and last-plus-one file to load, relative to the sorted
        filenames matching `path` and `ext`. Interpreted using python slice indexing conventions.

    recursive : boolean, optional, default=False
        If true, will recursively descend directories from path, loading all files
        with an extension matching 'ext'.

    nplanes : positive integer, optional, default=None
        If passed, will cause single files to be subdivided into nplanes separate images.
        Otherwise, each file is taken to represent one image.

    npartitions : int, optional, default=None
        Number of partitions for computational engine,
        if None will use default for engine.
    """
    import json
    from thunder.data.readers import get_file_reader, FileNotFoundError
    try:
        reader = get_file_reader(path)(credentials=credentials())
        buf = reader.read(path, filename=conf)
        params = json.loads(buf)
    except FileNotFoundError:
        params = {}

    if 'dtype' in params.keys():
        dtype = params['dtype']
    if 'dims' in params.keys():
        dims = params['dims']

    if not dims:
        raise ValueError("Image dimensions must be specified as argument or in a conf.json file")

    if not dtype:
        dtype = 'int16'

    if nplanes is not None:
        if nplanes <= 0:
            raise ValueError("nplanes must be positive if passed, got %d" % nplanes)
        if dims[-1] % nplanes:
            raise ValueError("Last dimension '%d' must be divisible by nplanes '%d'" %
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

def fromtif(path, ext='tif', start=None, stop=None, recursive=False,
            nplanes=None, npartitions=None):
    """
    Loads images from single or multi-page TIF files.

    Parameters
    ----------
    path : str
        Path to data files or directory, specified as either a local filesystem path
        or in a URI-like format, including scheme. May include a single '*' wildcard character.

    ext : string, optional, default="tif"
        Extension required on data files to be loaded.

    start, stop : nonnegative int, optional, default=None
        Indices of the first and last-plus-one file to load, relative to the sorted
        filenames matching `path` and `ext`. Interpreted using python slice indexing conventions.

    recursive : boolean, optional, default=False
        If true, will recursively descend directories from path, loading all files
        with an extension matching 'ext'.

    nplanes : positive integer, optional, default=None
        If passed, will cause single files to be subdivided into nplanes separate images.
        Otherwise, each file is taken to represent one image.

    npartitions : int, optional, default=None
        Number of partitions for computational engine,
        if None will use default for engine.
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
    """
    Load images from PNG files.

    Parameters
    ----------
    path : str
        Path to data files or directory, specified as either a local filesystem path
        or in a URI-like format, including scheme. May include a single '*' wildcard character.

    ext : string, optional, default="tif"
        Extension required on data files to be loaded.

    start, stop : nonnegative int, optional, default=None
        Indices of the first and last-plus-one file to load, relative to the sorted
        filenames matching `path` and `ext`. Interpreted using python slice indexing conventions.

    recursive : boolean, optional, default=False
        If true, will recursively descend directories from path, loading all files
        with an extension matching 'ext'.

    npartitions : int, optional, default=None
        Number of partitions for computational engine,
        if None will use default for engine.
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
        # Data is a 4d numpy array in t,z,y,x order. Swap axes and remove time dimension
        # to give back a 3d numpy array in x,y,z order
        data = swapaxes(data[0, :, :, :], 0, 2)

        return data

    return fromurls(enumerate(urlList), accessor=read, npartitions=len(urlList))

def fromrandom(shape=(10, 50, 50), npartitions=1, seed=42):
    """
    Generate random image data.

    Parameters
    ----------
    shape : tuple, optional, default=(10, 50, 50)
        Dimensions of images.

    npartitions : int, optional, default=1
        Number of partitions.

    seed : int, optional, default=42
        Random seed.
    """
    seed = hash(seed)

    def generate(v):
        random.seed(seed + v)
        return random.randn(*shape[1:])

    return fromlist(range(shape[0]), accessor=generate, npartitions=npartitions)

def fromexample(name=None):
    """
    Load example image data.

    Data must be downloaded from S3, so this method requires
    an internet connection.

    Parameters
    ----------
    name : str
        Name of dataset, options include 'mouse' | 'fish.
        If not specified will print options.
    """
    datasets = ['mouse', 'fish']

    if name is None:
        print 'Availiable example datasets'
        for d in datasets:
            print '- ' + d
        return

    checkist.opts(name, datasets)

    if mode() == 'spark':

        path = 's3n://thunder-sample-data/images/' + name

        if name == 'mouse':
            data = frombinary(path=path, npartitions=1, order='F')

        if name == 'fish':
            data = fromtif(path=path, npartitions=1)

        data.cache()
        data.count()
        return data

    else:
        raise NotImplementedError("Loading not implemented for '%s' mode" % mode())
from numpy import array, arange, frombuffer, load, ndarray, asarray, random

from thunder import engine, mode, credentials


def fromrdd(rdd, **kwargs):
    from .series import Series
    return Series(rdd, **kwargs)

def fromlist(items, accessor=None, keys=None, npartitions=None, index=None, **kwargs):

    if mode() == 'spark':
        nrecords = len(items)
        if not keys:
            keys = map(lambda k: (k, ), range(len(items)))
        if not npartitions:
            npartitions = engine().defaultParallelism
        items = zip(keys, items)
        rdd = engine().parallelize(items, npartitions)
        if accessor:
            rdd = rdd.mapValues(accessor)
        return fromrdd(rdd, nrecords=nrecords, index=index, **kwargs)

    else:
        raise NotImplementedError("Loading not implemented for '%s' mode" % mode())

def fromarray(arrays, npartitions=None, index=None, keys=None):
    """
    Create a Series object from a sequence of 1d numpy arrays.
    """
    # recast singleton
    if isinstance(arrays, list):
        arrays = asarray(arrays)

    if isinstance(arrays, ndarray) and arrays.ndim > 1:
        arrays = list(arrays)

    # check shape and dtype
    shape = arrays[0].shape
    dtype = arrays[0].dtype
    for ary in arrays:
        if not ary.shape == shape:
            raise ValueError("Inconsistent array shapes: first array had shape %s, but other array has shape %s" %
                             (str(shape), str(ary.shape)))
        if not ary.dtype == dtype:
            raise ValueError("Inconsistent array dtypes: first array had dtype %s, but other array has dtype %s" %
                             (str(dtype), str(ary.dtype)))

    return fromlist(arrays, keys=keys, npartitions=npartitions, dtype=str(dtype), index=index)

def frommat(path, var, npartitions=None, keyFile=None, index=None):
    """
    Loads Series data stored in a Matlab .mat file.

    `datafile` must refer to a path visible to all workers, such as on NFS or similar mounted shared filesystem.
    """
    from scipy.io import loadmat
    data = loadmat(path)[var]
    if data.ndim > 2:
        raise IOError('Input data must be one or two dimensional')
    if keyFile:
        keys = map(lambda x: tuple(x), loadmat(keyFile)['keys'])
    else:
        keys = None

    return fromlist(data, keys=keys, npartitions=npartitions, dtype=str(data.dtype), index=index)

def fromnpy(path, npartitions=None, keyfile=None, index=None):
    """
    Loads Series data stored in the numpy save() .npy format.

    `datafile` must refer to a path visible to all workers, such as on NFS or similar mounted shared filesystem.
    """
    data = load(path)
    if data.ndim > 2:
        raise IOError('Input data must be one or two dimensional')
    if keyfile:
        keys = map(lambda x: tuple(x), load(keyfile))
    else:
        keys = None

    return fromlist(data, keys=keys, npartitions=npartitions, dtype=str(data.dtype), index=index)

def fromtext(path, npartitions=None, nkeys=None, ext="txt", dtype='float64'):
    """
    Loads Series data from text files.

    Parameters
    ----------
    dataPath : string
        Specifies the file or files to be loaded. dataPath may be either a URI (with scheme specified) or a path
        on the local filesystem.
        If a path is passed (determined by the absence of a scheme component when attempting to parse as a URI),
        and it is not already a wildcard expression and does not end in <ext>, then it will be converted into a
        wildcard pattern by appending '/*.ext'. This conversion can be avoided by passing a "file://" URI.

    dtype: dtype or dtype specifier, default 'float64'

    """
    if mode() == 'spark':
        from thunder.data.fileio.readers import normalizeScheme
        path = normalizeScheme(path, ext)

        def parse(line, nkeys_):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys_:], dtype=dtype)
            keys = tuple(int(x) for x in vec[:nkeys_])
            return keys, ts

        lines = engine().textFile(path, npartitions)
        data = lines.map(lambda x: parse(x, nkeys))
        return fromrdd(data, dtype=str(dtype))

    else:
        raise NotImplementedError("Loading not implemented for '%s' mode" % mode())

def frombinary(path, ext='bin', confFilename='conf.json', nkeys=None, nvalues=None,
               keyType=None, valueType=None, newDtype='smallfloat', casting='safe'):
    """
    Load a Series object from a directory of binary files.

    Parameters
    ----------
    dataPath : string URI or local filesystem path
        Specifies the directory or files to be loaded. May be formatted as a URI string with scheme (e.g. "file://",
        "s3n://", or "gs://"). If no scheme is present, will be interpreted as a path on the local filesystem. This path
        must be valid on all workers. Datafile may also refer to a single file, or to a range of files specified
        by a glob-style expression using a single wildcard character '*'.

    newDtype : dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
        Numpy dtype of output series data. Most methods expect Series data to be floating-point. Input data will be
        cast to the requested `newdtype` if not None - see Data `astype()` method.

    casting : 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
        Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

    maxPartitionSize : str, optional, default = '32mb'
        Maximum size of partitions as Java-style memory, will indirectly control the number of partitions

    """
    params = binaryconfig(path, confFilename, nkeys, nvalues, keyType, valueType)

    from thunder.data.fileio.readers import normalizeScheme
    dataPath = normalizeScheme(path, ext)

    from numpy import dtype as dtypeFunc
    keytype = dtypeFunc(params.keytype)
    valuetype = dtypeFunc(params.valuetype)

    keySize = params.nkeys * keytype.itemsize
    recordSize = keySize + params.nvalues * valuetype.itemsize

    if mode() == 'spark':
        lines = engine().binaryRecords(dataPath, recordSize)

        def get(kv):
            k = tuple(int(x) for x in frombuffer(buffer(kv, 0, keySize), dtype=keytype))
            v = frombuffer(buffer(kv, keySize), dtype=valuetype)
            return (k, v) if keySize > 0 else v

        raw = lines.map(get)
        if keySize == 0:
            raw = raw.zipWithIndex().map(lambda (v, k): ((k,), v))
        data = fromrdd(raw, dtype=str(valuetype), index=arange(params.nvalues))
        return data.astype(newDtype, casting)

    else:
        raise NotImplementedError("Loading not implemented for '%s' mode" % mode())

def binaryconfig(path, conf, nkeys, nvalues, keytype, valuetype):
    """
    Collects parameters to use for binary series loading.
    """
    import json
    from collections import namedtuple
    from thunder.data.fileio.readers import getFileReaderForPath, FileNotFoundError

    Parameters = namedtuple('BinaryLoadParameters', 'nkeys nvalues keytype valuetype')
    Parameters.__new__.__defaults__ = (None, None, 'int16', 'int16')

    reader = getFileReaderForPath(path)(credentials=credentials())
    try:
        buf = reader.read(path, filename=conf)
        params = json.loads(buf)
    except FileNotFoundError:
        params = {}

    for k in params.keys():
        if k not in Parameters._fields:
            del params[k]
    keywords = {'nkeys': nkeys, 'nvalues': nvalues, 'keytype': keytype, 'valuetype': valuetype}
    for k, v in keywords.items():
        if not v and not v == 0:
            del keywords[k]
    params.update(keywords)
    params = Parameters(**params)

    missing = []
    for name, val in params._asdict().iteritems():
        if not val and not val == 0:
            missing.append(name)
    if missing:
        raise ValueError("Missing parameters to load binary series files - " +
                         "these must be given either as arguments or in a configuration file: " +
                         str(tuple(missing)))
    return params

def fromrandom(shape=(100, 10), npartitions=1, seed=42):
    """
    Generate gaussian random Series data.

    Parameters
    ----------
    shape : tuple
        Dimensions of data

    npartitions : int
        Number of partitions with which to distribute data

    seed : int
        Randomization seed
    """
    seed = hash(seed)

    def generate(v):
        random.seed(seed + v)
        return random.randn(shape[1])

    return fromlist(range(shape[0]), accessor=generate, npartitions=npartitions)

def fromexample(name=None):
    """
    Load example Series data.

    Parameters
    ----------
    name : str
        Options include 'iris' | 'mouse' | 'fish. If not specified will print options.
    """
    import os
    import tempfile
    import shutil
    import checkist
    from boto.s3.connection import S3Connection

    datasets = ['iris', 'mouse', 'fish']

    if name is None:
        print 'Availiable example datasets'
        for d in datasets:
            print '- ' + d
        return

    checkist.opts(name, datasets)

    if mode() == 'spark':

        d = tempfile.mkdtemp()
        try:
            os.mkdir(os.path.join(d, 'series'))
            os.mkdir(os.path.join(d, 'series', name))
            conn = S3Connection(anon=True)
            bucket = conn.get_bucket('thunder-sample-data')
            for key in bucket.list(os.path.join('series', name)):
                if not key.name.endswith('/'):
                    key.get_contents_to_filename(os.path.join(d, key.name))
            data = frombinary(os.path.join(d, 'series', name))
            data.cache()
            data.count()
        finally:
            shutil.rmtree(d)

        return data

    else:
        raise NotImplementedError("Loading not implemented for '%s' mode" % mode())
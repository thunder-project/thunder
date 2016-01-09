from numpy import array, arange, frombuffer, load, asarray, random, maximum, \
    fromstring, expand_dims

from ..utils import check_spark
spark = check_spark()


def fromrdd(rdd, nrecords=None, shape=None, index=None, dtype=None):
    """
    Load Series object from a Spark RDD.

    Assumes keys are tuples with increasing and unique indices,
    and values are 1d ndarrays. Will try to infer properties
    that are not explicitly provided.

    Parameters
    ----------
    rdd : SparkRDD
        An RDD containing series data.

    shape : tuple or array, optional, default = None
        Total shape of data (if provided will avoid check).

    nrecords : int, optional, default = None
        Number of records (if provided will avoid check).

    index : array, optional, default = None
        Index for records, if not provided will use (0, 1, ...)

    dtype : string, default = None
       Data numerical type (if provided will avoid check)
    """
    from .series import Series
    from bolt.spark.array import BoltArraySpark

    if index is None or dtype is None:
        item = rdd.values().first()

    if index is None:
        index = range(len(item))

    if dtype is None:
        dtype = item.dtype

    if shape is None or nrecords is None:
        nrecords = rdd.count()

    if shape is None:
        shape = (nrecords, asarray(index).shape[0])

    values = BoltArraySpark(rdd, shape=shape, dtype=dtype, split=1)
    return Series(values, index=index)

def fromarray(values, index=None, npartitions=None, engine=None):
    """
    Load Series object from a local numpy array.

    Assumes that all but final dimension index the records,
    and the size of the final dimension is the length of each record,
    e.g. a (2, 3, 4) array will be treated as 2 x 3 records of size (4,)

    Parameters
    ----------
    values : array-like
        An array containing the data.

    index : array, optional, default = None
        Index for records, if not provided will use (0,1,...,N)
        where N is the length of each record.

    npartitions : int, default = None
        Number of partitions for parallelization (Spark only)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)
    """
    from .series import Series
    import bolt

    values = asarray(values)

    if values.ndim < 2:
        values = expand_dims(values, 0)

    if index is not None and not asarray(index).shape[0] == values.shape[-1]:
        raise ValueError('Index length %s not equal to record length %s'
                         % (asarray(index).shape[0], values.shape[-1]))
    if index is None:
        index = arange(values.shape[-1])

    if spark and isinstance(engine, spark):
        axis = tuple(range(values.ndim - 1))
        values = bolt.array(values, context=engine, npartitions=npartitions, axis=axis)
        return Series(values, index=index)

    return Series(values, index=index)

def fromlist(items, accessor=None, index=None, dtype=None, npartitions=None, engine=None):
    """
    Create a Series object from a list of items and optional accessor function.

    Will call accessor function on each item from the list,
    providing a generic interface for data loading.

    Parameters
    ----------
    items : list
        A list of items to load.

    accessor : function, optional, default = None
        A function to apply to each item in the list during loading.

    index : array, optional, default = None
        Index for records, if not provided will use (0,1,...,N)
        where N is the length of each record.

    dtype : string, default = None
       Data numerical type (if provided will avoid check)

    npartitions : int, default = None
        Number of partitions for parallelization (Spark only)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)
    """
    if spark and isinstance(engine, spark):
        if dtype is None:
            dtype = accessor(items[0]).dtype if accessor else items[0].dtype
        nrecords = len(items)
        keys = map(lambda k: (k, ), range(len(items)))
        if not npartitions:
            npartitions = engine.defaultParallelism
        items = zip(keys, items)
        rdd = engine.parallelize(items, npartitions)
        if accessor:
            rdd = rdd.mapValues(accessor)
        return fromrdd(rdd, nrecords=nrecords, index=index, dtype=dtype)

    else:
        if accessor:
            items = [accessor(i) for i in items]
        return fromarray(items, index=index)

def frommat(path, var, index=None, npartitions=None, engine=None):
    """
    Loads Series data stored in a Matlab .mat file.

    Parameters
    ----------
    path : str
        Path to data file.

    var : str
        Variable name.

    index : array, optional, default = None
        Index for records, if not provided will use (0,1,...,N)
        where N is the length of each record.

    npartitions : int, default = None
        Number of partitions for parallelization (Spark only)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)
    """
    from scipy.io import loadmat
    data = loadmat(path)[var]
    if data.ndim > 2:
        raise IOError('Input data must be one or two dimensional')

    return fromarray(data, npartitions=npartitions, index=index, engine=engine)

def fromnpy(path,  index=None, npartitions=None, engine=None):
    """
    Loads Series data stored in the numpy save() .npy format.

    Parameters
    ----------
    path : str
        Path to data file.

    index : array, optional, default = None
        Index for records, if not provided will use (0,1,...,N)
        where N is the length of each record.

    npartitions : int, default = None
        Number of partitions for parallelization (Spark only)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)
    """
    data = load(path)
    if data.ndim > 2:
        raise IOError('Input data must be one or two dimensional')

    return fromarray(data, npartitions=npartitions, index=index, engine=engine)

def fromtext(path, npartitions=None, nkeys=None, ext='txt', dtype='float64',
             index=None, engine=None, credentials=None):
    """
    Loads Series data from text files.

    Assumes data are formatted as rows, where each record is a row
    of numbers separated by spaces, the first numbers in each row
    are keys, and the remaining numbers are values, e.g.
    'k v v v v'

    Parameters
    ----------
    path : string
        Directory to load from, can be a URI string with scheme
        (e.g. "file://", "s3n://", or "gs://"), or a single file,
        or a directory, or a directory with a single wildcard character.

    npartitions : int, default = None
        Number of partitions for parallelization (Spark only)

    nkeys : int, optional, default = None
        Number of keys per record.

    ext : str, optional, default = 'txt'
        File extension.

    dtype: dtype or dtype specifier, default 'float64'
        Numerical type to use for data after converting from text.

    index : array, optional, default = None
        Index for records, if not provided will use (0, 1, ...)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)

    credentials : dict, default = None
        Credentials for remote storage (e.g. S3) in the form {access: ***, secret: ***}
    """
    from thunder.readers import normalize_scheme, get_parallel_reader
    path = normalize_scheme(path, ext)

    if spark and isinstance(engine, spark):

        def parse(line, nkeys_):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys_:], dtype=dtype)
            keys = tuple(int(x) for x in vec[:nkeys_])
            return keys, ts

        lines = engine.textFile(path, npartitions)
        data = lines.map(lambda x: parse(x, nkeys))
        return fromrdd(data, dtype=str(dtype), index=index)

    else:
        reader = get_parallel_reader(path)(engine, credentials=credentials)
        data = reader.read(path, ext=ext)

        values = []
        for kv in data:
            for line in kv[1].split('\n')[:-1]:
                values.append(fromstring(line, sep=' '))

        values = asarray(values)

        if nkeys:
            basedims = tuple(asarray(values[:, 0:nkeys]).max(axis=0) + 1)
            nvalues = values.shape[-1] - nkeys
            values = values[:, nkeys:].reshape(basedims + (nvalues,))

        return fromarray(values, index=index)

def frombinary(path, ext='bin', conf='conf.json', nkeys=None, nvalues=None,
               keytype=None, valuetype=None, index=None, engine=None, credentials=None):
    """
    Load a Series object from flat binary files.

    Parameters
    ----------
    path : string URI or local filesystem path
        Directory to load from, can be a URI string with scheme
        (e.g. "file://", "s3n://", or "gs://"), or a single file,
        or a directory, or a directory with a single wildcard character.

    ext : str, optional, default = 'bin'
        Optional file extension specifier.

    conf : str, optional, default = 'conf.json'
        Name of conf file with type and size information.

    nkeys, nvalues : int, optional, default = None
        Parameters of binary data, can be specified here or in a configuration file.

    keytype, valuetype : str, optional, default = None
        Parameters of binary data, can be specified here or in a configuration file.

    index : array, optional, default = None
        Index for records, if not provided will use (0, 1, ...)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)

    credentials : dict, default = None
        Credentials for remote storage (e.g. S3) in the form {access: ***, secret: ***}
    """
    params = binaryconfig(path, conf, nkeys, nvalues, keytype, valuetype, credentials)

    from thunder.readers import normalize_scheme, get_parallel_reader
    path = normalize_scheme(path, ext)

    from numpy import dtype as dtypeFunc
    keytype = dtypeFunc(params.keytype)
    valuetype = dtypeFunc(params.valuetype)

    keysize = params.nkeys * keytype.itemsize
    valuesize = params.nvalues * valuetype.itemsize
    recordsize = keysize + valuesize

    if spark and isinstance(engine, spark):
        lines = engine.binaryRecords(path, recordsize)

        def get(kv):
            k = tuple(int(x) for x in frombuffer(buffer(kv, 0, keysize), dtype=keytype))
            v = frombuffer(buffer(kv, keysize), dtype=valuetype)
            return (k, v) if keysize > 0 else v

        raw = lines.map(get)
        if keysize == 0:
            raw = raw.zipWithIndex().map(lambda (v, k): ((k,), v))
        shape = tuple(raw.keys().reduce(maximum) + 1) + (params.nvalues,)

        if not index:
            index = arange(params.nvalues)

        return fromrdd(raw, dtype=str(valuetype), shape=shape, index=index)

    else:
        reader = get_parallel_reader(path)(engine, credentials=credentials)
        data = reader.read(path, ext=ext)

        keys = []
        values = []
        for kv in data:
            buf = kv[1]
            offset = 0
            while offset < len(buf):
                k = frombuffer(buffer(buf, offset, keysize), dtype=keytype)
                v = frombuffer(buffer(buf, offset + keysize, valuesize), dtype=valuetype)
                keys.append(k)
                values.append(v)
                offset += recordsize

        values = asarray(values)
        if keysize == 0:
            basedims = (values.shape[0],)
        else:
            basedims = tuple(asarray(keys).max(axis=0) + 1)
        values = values.reshape(basedims + (params.nvalues,))

        return fromarray(values, index=index)

def binaryconfig(path, conf, nkeys, nvalues, keytype, valuetype, credentials):
    """
    Collects parameters to use for binary series loading.
    """
    import json
    from collections import namedtuple
    from thunder.readers import get_file_reader, FileNotFoundError

    Parameters = namedtuple('BinaryLoadParameters', 'nkeys nvalues keytype valuetype')
    Parameters.__new__.__defaults__ = (None, None, 'int16', 'int16')

    reader = get_file_reader(path)(credentials=credentials)
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

def fromrandom(shape=(100, 10), npartitions=1, seed=42, engine=None):
    """
    Generate gaussian random series data.

    Parameters
    ----------
    shape : tuple
        Dimensions of data.

    npartitions : int
        Number of partitions with which to distribute data.

    seed : int
        Randomization seed.
    """
    seed = hash(seed)

    def generate(v):
        random.seed(seed + v)
        return random.randn(shape[1])

    return fromlist(range(shape[0]), accessor=generate, npartitions=npartitions, engine=engine)

def fromexample(name=None, engine=None):
    """
    Load example series data.

    Data must be downloaded from S3, so this method requires
    an internet connection.

    Parameters
    ----------
    name : str
        Name of dataset, options include 'iris' | 'mouse' | 'fish'.
        If not specified will print options.
    """
    import os
    import tempfile
    import shutil
    import checkist
    from boto.s3.connection import S3Connection

    datasets = ['iris', 'mouse', 'fish']

    if name is None:
        print 'Availiable example series datasets'
        for d in datasets:
            print '- ' + d
        return

    checkist.opts(name, datasets)

    d = tempfile.mkdtemp()

    try:
        os.mkdir(os.path.join(d, 'series'))
        os.mkdir(os.path.join(d, 'series', name))
        conn = S3Connection(anon=True)
        bucket = conn.get_bucket('thunder-sample-data')
        for key in bucket.list(os.path.join('series', name)):
            if not key.name.endswith('/'):
                key.get_contents_to_filename(os.path.join(d, key.name))
        data = frombinary(os.path.join(d, 'series', name), engine=engine)

        if spark and isinstance(engine, spark):
            data.cache()
            data.compute()

    finally:
        shutil.rmtree(d)

    return data

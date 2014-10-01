import glob
import os
import urllib
import urlparse
import errno

_have_boto = False
try:
    import boto
    _have_boto = True
except ImportError:
    pass


class FileNotFoundError(IOError):
    """An exception to be thrown when reader implementations can't find a requested file.

    Implementations are responsible for watching for their own appropriate exceptions and rethrowing
    FileNotFoundError.

    See PEP 3151 for background and inspiration.
    """
    pass


def _localRead(filepath):
    """Wrapper around open(filepath, 'rb') that returns the contents of the file as a string.

    Will rethrow FileNotFoundError if it receives an IOError with error number indicating that the file isn't found.
    """
    buf = None
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
    except IOError, e:
        if e.errno == errno.ENOENT:
            raise FileNotFoundError(e)
        else:
            raise
    return buf


class LocalFSParallelReader(object):
    def __init__(self, sparkcontext):
        self.sc = sparkcontext
        self.lastnrecs = None

    @staticmethod
    def uriToPath(uri):
        # thanks stack overflow:
        # http://stackoverflow.com/questions/5977576/is-there-a-convenient-way-to-map-a-file-uri-to-os-path
        path = urllib.url2pathname(urlparse.urlparse(uri).path)
        if uri and (not path):
            # passed a nonempty uri, got an empty path back
            # this happens when given a file uri that starts with "file://" instead of "file:///"
            # error here to prevent unexpected behavior of looking at current working directory
            raise ValueError("Could not interpret %s as URI. Note absolute paths in URIs should start with 'file:///', not 'file://'")
        return path

    @staticmethod
    def listFiles(abspath, ext=None, startidx=None, stopidx=None):

        if os.path.isdir(abspath):
            if ext:
                files = sorted(glob.glob(os.path.join(abspath, '*.' + ext)))
            else:
                files = sorted(os.listdir(abspath))
        else:
            files = sorted(glob.glob(abspath))

        if len(files) < 1:
            raise IOError('cannot find files of type "%s" in %s' % (ext if ext else '*', abspath))

        if startidx or stopidx:
            if startidx is None:
                startidx = 0
            if stopidx is None:
                stopidx = len(files)
            files = files[startidx:stopidx]

        return files

    def read(self, datapath, ext=None, startidx=None, stopidx=None):
        """Returns RDD of int, buffer k/v pairs
        """
        abspath = self.uriToPath(datapath)
        filepaths = self.listFiles(abspath, ext=ext, startidx=startidx, stopidx=stopidx)

        lfilepaths = len(filepaths)
        self.lastnrecs = lfilepaths
        return self.sc.parallelize(enumerate(filepaths), lfilepaths).map(lambda (k, v): (k, _localRead(v)))


class _BotoS3Client(object):
    # todo: boto s3 readers should throw FileNotFoundError as appropriate
    @staticmethod
    def _parseS3Schema(datapath):
        parseresult = urlparse.urlparse(datapath)
        return parseresult.netloc, parseresult.path.lstrip("/")

    def __init__(self):
        if not _have_boto:
            raise ValueError("The boto package does not appear to be available; boto is required for BotoS3Reader")
        if (not 'AWS_ACCESS_KEY_ID' in os.environ) or (not 'AWS_SECRET_ACCESS_KEY' in os.environ):
            raise ValueError("The environment variables 'AWS_ACCESS_KEY_ID' and 'AWS_SECRET_ACCESS_KEY' must be set in order to read from s3")

        # save keys in this object and serialize out to workers to prevent having to set env vars separately on all
        # nodes in the cluster
        self._access_key = os.environ['AWS_ACCESS_KEY_ID']
        self._secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

    @property
    def accessKey(self):
        return self._access_key

    @property
    def secretKey(self):
        return self._secret_key


class BotoS3ParallelReader(_BotoS3Client):
    def __init__(self, sparkcontext):
        super(BotoS3ParallelReader, self).__init__()
        self.sc = sparkcontext
        self.lastnrecs = None

    def _listFiles(self, datapath, ext=None, startidx=None, stopidx=None):
        bucketname, keyname = _BotoS3Client._parseS3Schema(datapath)
        conn = boto.connect_s3(aws_access_key_id=self.accessKey, aws_secret_access_key=self.secretKey)
        bucket = conn.get_bucket(bucketname)
        keylist = bucket.list(prefix=keyname)
        if ext:
            keynamelist = [key.name for key in keylist if key.name.endswith(ext)]
        else:
            keynamelist = [key.name for key in keylist]
        keynamelist.sort()
        if startidx or stopidx:
            if startidx is None:
                startidx = 0
            if stopidx is None:
                stopidx = len(keynamelist)
            keynamelist = keynamelist[startidx:stopidx]
        return bucketname, keynamelist

    def read(self, datapath, ext=None, startidx=None, stopidx=None):
        bucketname, keynamelist = self._listFiles(datapath, ext=ext, startidx=startidx, stopidx=stopidx)

        access_key = self.accessKey
        secret_key = self.secretKey

        def readSplitFromS3(kvIter):
            conn = boto.connect_s3(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
            # bucket = conn.get_bucket(bucketname, validate=False)
            bucket = conn.get_bucket(bucketname)
            for kv in kvIter:
                idx, keyname = kv
                #key = bucket.get_key(keyname, validate=False)
                key = bucket.get_key(keyname)
                buf = key.get_contents_as_string()
                yield idx, buf

        # don't specify number of splits here - allow reuse of connections within partition
        self.lastnrecs = len(keynamelist)
        return self.sc.parallelize(enumerate(keynamelist)).mapPartitions(readSplitFromS3)


class LocalFSFileReader(object):
    def read(self, datapath, filename=None):
        abspath = LocalFSParallelReader.uriToPath(datapath)
        if filename:
            abspath = os.path.join(abspath, filename)

        return _localRead(abspath)


class BotoS3FileReader(_BotoS3Client):
    def read(self, datapath, filename=None):
        bucketname, keyname = _BotoS3Client._parseS3Schema(datapath)

        if filename:
            if not keyname.endswith("/"):
                keyname += "/"
            keyname = keyname + filename

        conn = boto.connect_s3(aws_access_key_id=self.accessKey, aws_secret_access_key=self.secretKey)
        bucket = conn.get_bucket(bucketname)
        return bucket.get_key(keyname).get_contents_as_string()


SCHEMAS_TO_PARALLELREADERS = {
    '': LocalFSParallelReader,
    'file': LocalFSParallelReader,
    's3': BotoS3ParallelReader,
    's3n': BotoS3ParallelReader,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}

SCHEMAS_TO_FILEREADERS = {
    '': LocalFSFileReader,
    'file': LocalFSFileReader,
    's3': BotoS3FileReader,
    's3n': BotoS3FileReader,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}


def getByScheme(datapath, lookup, default):
    parseresult = urlparse.urlparse(datapath)
    clazz = lookup.get(parseresult.scheme, default)
    if clazz is None:
        raise NotImplementedError("No implementation for scheme " + parseresult.scheme)
    return clazz


def getParallelReaderForPath(datapath):
    return getByScheme(datapath, SCHEMAS_TO_PARALLELREADERS, LocalFSParallelReader)


def getFileReaderForPath(datapath):
    return getByScheme(datapath, SCHEMAS_TO_FILEREADERS, LocalFSFileReader)
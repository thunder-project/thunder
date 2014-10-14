import glob
import os
import urllib
import urlparse
import errno
import itertools

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


def appendExtensionToPathSpec(datapath, ext=None):
    if ext:
        if '*' in datapath:
            if datapath[-1] == '*':
                # path ends in wildcard but without postfix
                # use ext as postfix
                return datapath + ext
            else:
                # ext specified, but datapath apparently already has a postfix
                # drop ext and use existing postfix
                return datapath
        else:
            # no wildcard in path yet
            return datapath+'*'+ext
    else:
        return datapath


def selectByStartAndStopIndices(files, startidx, stopidx):
    if startidx or stopidx:
        if startidx is None:
            startidx = 0
        if stopidx is None:
            stopidx = len(files)
        files = files[startidx:stopidx]
    return files


def _localRead(filepath, startOffset=None, size=-1):
    """Wrapper around open(filepath, 'rb') that returns the contents of the file as a string.

    Will rethrow FileNotFoundError if it receives an IOError with error number indicating that the file isn't found.
    """
    buf = None
    try:
        with open(filepath, 'rb') as f:
            if startOffset:
                f.seek(startOffset)
            buf = f.read(size)
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
            raise FileNotFoundError('cannot find files of type "%s" in %s' % (ext if ext else '*', abspath))

        files = selectByStartAndStopIndices(files, startidx, stopidx)

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
    def parseS3Query(query, delim='/'):
        keyname = ''
        prefix = ''
        postfix = ''

        parseresult = urlparse.urlparse(query)
        bucketname = parseresult.netloc
        keyquery = parseresult.path.lstrip(delim)

        if not parseresult.scheme.lower() in ('', "s3", "s3n"):
            raise ValueError("Query scheme must be one of '', 's3', or 's3n'; got: '%s'" % parseresult.scheme)

        # special case handling for strings of form "/bucket/dir":
        if (not bucketname.strip()) and keyquery:
            toks = keyquery.split(delim, 1)
            bucketname = toks[0]
            if len(toks) == 2:
                keyquery = toks[1]
            else:
                keyquery = ''

        if not bucketname.strip():
            raise ValueError("Could not parse bucket name from query string '%s'" % query)

        keytoks = keyquery.split("*")
        nkeytoks = len(keytoks)
        if nkeytoks == 0:
            pass
        elif nkeytoks == 1:
            keyname = keytoks[0]
        elif nkeytoks == 2:
            rdelimidx = keytoks[0].rfind(delim)
            if rdelimidx >= 0:
                keyname = keytoks[0][:(rdelimidx+1)]
                prefix = keytoks[0][(rdelimidx+1):] if len(keytoks[0]) > (rdelimidx+1) else ''
            else:
                prefix = keytoks[0]
            postfix = keytoks[1]
        else:
            raise ValueError("Only one wildcard ('*') allowed in query string, got: '%s'" % query)

        return bucketname, keyname, prefix, postfix

    @staticmethod
    def checkPrefix(bucket, keypath, delim='/'):
        return len(bucket.get_all_keys(prefix=keypath, delimiter=delim, max_keys=1)) > 0

    @staticmethod
    def filterPredicate(key, post, inclusive=False):
        kname = key.name
        retval = not inclusive
        if kname.endswith(post):
            retval = not retval

        return retval

    @staticmethod
    def retrieveKeys(bucket, key, prefix='', postfix='', delim='/', exclude_directories=True):
        if key and prefix:
            assert key.endswith(delim)

        keypath = key+prefix
        # if we are asking for a key that doesn't end in a delimiter, check whether it might
        # actually be a directory
        if not keypath.endswith(delim) and keypath:
            # not all directories have actual keys associated with them
            # check for matching prefix instead of literal key:
            if _BotoS3Client.checkPrefix(bucket, keypath+delim, delim=delim):
                # found a directory; change path so that it explicitly refers to directory
                keypath += delim

        results = bucket.list(prefix=keypath, delimiter=delim)
        if postfix:
            return itertools.ifilter(lambda k_: _BotoS3Client.filterPredicate(k_, postfix, inclusive=True), results)
        elif exclude_directories:
            return itertools.ifilter(lambda k_: _BotoS3Client.filterPredicate(k_, delim, inclusive=False), results)
        else:
            return results

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

    def _listFiles(self, datapath, startidx=None, stopidx=None):
        parse = _BotoS3Client.parseS3Query(datapath)

        conn = boto.connect_s3(aws_access_key_id=self.accessKey, aws_secret_access_key=self.secretKey)
        bucket = conn.get_bucket(parse[0])
        keys = _BotoS3Client.retrieveKeys(bucket, parse[1], prefix=parse[2], postfix=parse[3])
        keynamelist = [key.name for key in keys]
        keynamelist.sort()

        keynamelist = selectByStartAndStopIndices(keynamelist, startidx, stopidx)

        return bucket.name, keynamelist

    def read(self, datapath, ext=None, startidx=None, stopidx=None):
        datapath = appendExtensionToPathSpec(datapath, ext)
        bucketname, keynamelist = self._listFiles(datapath, startidx=startidx, stopidx=stopidx)

        if not keynamelist:
            raise FileNotFoundError("No S3 objects found for '%s'" % datapath)

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

    def list(self, datapath, filename=None):
        abspath = LocalFSParallelReader.uriToPath(datapath)

        if filename:
            if os.path.isdir(abspath):
                abspath = os.path.join(abspath, filename)
            else:
                abspath = os.path.join(os.path.dirname(abspath), filename)

        return sorted(glob.glob(abspath))

    def read(self, datapath, filename=None, startOffset=None, size=-1):
        filenames = self.list(datapath, filename=filename)

        if not filenames:
            raise FileNotFoundError("No file found matching: '%s'" % datapath)
        if len(filenames) > 1:
            raise ValueError("Found multiple files matching: '%s'" % datapath)

        return _localRead(filenames[0], startOffset=startOffset, size=size)

    def open(self, datapath, filename=None):
        filenames = self.list(datapath, filename=filename)

        if not filenames:
            raise FileNotFoundError("No file found matching: '%s'" % datapath)
        if len(filenames) > 1:
            raise ValueError("Found multiple files matching: '%s'" % datapath)

        return open(filenames[0], 'rb')


class BotoS3FileReader(_BotoS3Client):

    def __getMatchingKeys(self, datapath, filename=None):
        parse = _BotoS3Client.parseS3Query(datapath)
        conn = boto.connect_s3(aws_access_key_id=self.accessKey, aws_secret_access_key=self.secretKey)
        bucketname = parse[0]
        keyname = parse[1]
        bucket = conn.get_bucket(bucketname)

        if filename:
            # check whether last section of datapath refers to a directory
            if not keyname.endswith("/"):
                if self.checkPrefix(bucket, keyname + "/"):
                    # keyname is a directory, but we've omitted the trailing "/"
                    keyname += "/"
                else:
                    # assume keyname refers to an object other than a directory
                    # look for filename in same directory as keyname
                    slidx = keyname.rfind("/")
                    if slidx >= 0:
                        keyname = keyname[:(slidx+1)]
                    else:
                        # no directory separators, so our object is in the top level of the bucket
                        keyname = ""
            keyname += filename

        return _BotoS3Client.retrieveKeys(bucket, keyname)

    def list(self, datapath, filename=None):
        keys = self.__getMatchingKeys(datapath, filename=filename)
        keynames = [key.bucket.name + "/" + key.name for key in keys]
        return sorted(keynames)

    def __getSingleMatchingKey(self, datapath, filename=None):
        keys = self.__getMatchingKeys(datapath, filename=filename)
        # keys is probably a lazy-loading ifilter iterable
        try:
            key = keys.next()
        except StopIteration:
            raise FileNotFoundError("Could not find S3 object for: '%s'" % datapath)

        # we expect to only have a single key returned
        nextkey = None
        try:
            nextkey = keys.next()
        except StopIteration:
            pass
        if nextkey:
            raise ValueError("Found multiple S3 keys for: '%s'" % datapath)
        return key

    def read(self, datapath, filename=None, startOffset=None, size=-1):
        key = self.__getSingleMatchingKey(datapath, filename=filename)

        if startOffset or (size > -1):
            # specify Range header in S3 request
            # see: http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.35
            # and: http://docs.aws.amazon.com/AmazonS3/latest/API/RESTObjectGET.html
            if not startOffset:
                startOffset = 0
            if size > -1:
                sizestr = startOffset + size - 1  # range header is inclusive
            else:
                sizestr = ""
            hdrs = {"Range": "bytes=%d-%s" % (startOffset, sizestr)}
            return key.get_contents_as_string(headers=hdrs)
        else:
            return key.get_contents_as_string()

    def open(self, datapath, filename=None):
        key = self.__getSingleMatchingKey(datapath, filename=filename)
        return BotoS3ReadFileHandle(key)


class BotoS3ReadFileHandle(object):
    def __init__(self, key):
        self._key = key
        self._closed = False
        self._offset = 0

    def close(self):
        self._key.close(fast=True)
        self._closed = True

    def read(self, size=-1):
        if self._offset or (size > -1):
            if size > -1:
                sizestr = str(self._offset + size - 1)  # range header is inclusive
            else:
                sizestr = ""
            hdrs = {"Range": "bytes=%d-%s" % (self._offset, sizestr)}
        else:
            hdrs = {}
        buf = self._key.get_contents_as_string(headers=hdrs)
        self._offset += len(buf)
        return buf

    def seek(self, offset, whence=0):
        if whence == 0:
            self._offset = offset
        elif whence == 1:
            self._offset += offset
        elif whence == 2:
            self._offset = self._key.size + offset
        else:
            raise IOError("Invalid 'whence' argument, must be 0, 1, or 2. See file().seek.")

    def tell(self):
        return self._offset

    @property
    def closed(self):
        return self._closed

    @property
    def name(self):
        return self._key.bucket.name + "/" + self._key.name

    @property
    def mode(self):
        return "rb"



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
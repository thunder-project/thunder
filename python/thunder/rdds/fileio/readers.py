"""Classes that abstract reading from various types of filesystems.

Currently two types of 'filesystem' are supported:

* the local file system, via python's native file() objects

* Amazon's S3, using the boto library (only if boto is installed; boto is not a requirement)

For each filesystem, two types of reader classes are provided:

* parallel readers are intended to serve as the entry point to a Spark workflow. They provide a read() method
that itself calls the spark context's parallelize() method, setting up a workflow with one partition per file. This
method returns a Spark RDD of <string filename, string binary data>.

* file readers are intended to abstract across the supported filesystems, providing a consistent interface to several
common file and filesystem operations. These include listing files in a directory, reading the contents of a file,
and providing a file handle or handle-like object that itself supports read(), seek(), and tell() operations.

The reader classes also all support a common syntax for path specifications, including both "standard" file paths
and "URI-like" syntax with an explicitly specified scheme (for instance, "file://" or "s3n://"). This path specification
syntax allows a single wildcard "*" character in the filename, making possible paths like
"s3n:///my-bucket/key-one/foo*.bar", referring to "every object in the S3 bucket my-bucket whose key starts with
'key-one/foo' and ends with '.bar'".
"""
import errno
import fnmatch
import glob
import itertools
import os
import urllib
import urlparse

_have_boto = False
try:
    import boto
    _have_boto = True
except ImportError:
    boto = None


class FileNotFoundError(IOError):
    """An exception to be thrown when reader implementations can't find a requested file.

    Implementations are responsible for watching for their own appropriate exceptions and rethrowing
    FileNotFoundError.

    See PEP 3151 for background and inspiration.
    """
    pass


def appendExtensionToPathSpec(datapath, ext=None):
    """Helper function for consistent handling of paths given with separately passed file extensions

    Returns
    -------
    result: string datapath
        datapath string formed by concatenating passed `datapath` with "*" and passed `ext`, with some
        normalization as appropriate
    """
    if ext:
        if '*' in datapath:
            # we already have a literal wildcard, which we take as a sign that the user knows
            # what they're doing and don't want us overriding their path by appending extensions to it
            return datapath
        elif os.path.splitext(datapath)[1]:
            # looks like we already have a literal extension specified at the end of datapath.
            # go with that.
            return datapath
        else:
            # no wildcard in path yet
            # check whether we already end in `ext`, which suggests we've been passed a literal filename.
            # prepend '.' to ext, as mild protection against the case where we have a directory 'bin' and
            # are looking in it for files named '*.bin'.
            if not ext.startswith('.'):
                ext = '.'+ext
            if not datapath.endswith(ext):
                # we have an extension and we'd like to append it.
                # we assume that datapath should be pointing to a directory at this point, but we might
                # or might not have a directory separator at the end of it. add it if we don't.
                if not datapath.endswith(os.path.sep):
                    datapath += os.path.sep
                # return a path with "/*."+`ext` added to it.
                return datapath+'*'+ext
            else:
                # we are asking to append `ext`, but it looks like datapath already ends with '.'+`ext`
                return datapath
    else:
        return datapath


def selectByStartAndStopIndices(files, startidx, stopidx):
    """Helper function for consistent handling of start and stop indices
    """
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
    """Parallel reader backed by python's native file() objects.
    """
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
            raise ValueError("Could not interpret %s as URI. " +
                             "Note absolute paths in URIs should start with 'file:///', not 'file://'")
        return path

    @staticmethod
    def _listFilesRecursive(abspath, ext=None):
        filenames = set()
        for root, dirs, files in os.walk(abspath):
            if ext:
                files = fnmatch.filter(files, '*.' + ext)
            for filename in files:
                filenames.add(os.path.join(root, filename))
        filenames = list(filenames)
        filenames.sort()
        return sorted(filenames)

    @staticmethod
    def _listFilesNonRecursive(abspath, ext=None):
        if os.path.isdir(abspath):
            if ext:
                files = glob.glob(os.path.join(abspath, '*.' + ext))
            else:
                files = [os.path.join(abspath, fname) for fname in os.listdir(abspath)]
        else:
            files = glob.glob(abspath)
        # filter out directories
        files = [fpath for fpath in files if not os.path.isdir(fpath)]
        return sorted(files)

    def listFiles(self, abspath, ext=None, startidx=None, stopidx=None, recursive=False):
        """Get sorted list of file paths matching passed `abspath` path and `ext` filename extension
        """
        files = LocalFSParallelReader._listFilesNonRecursive(abspath, ext) if not recursive else \
            LocalFSParallelReader._listFilesRecursive(abspath, ext)

        if len(files) < 1:
            raise FileNotFoundError('cannot find files of type "%s" in %s' % (ext if ext else '*', abspath))

        files = selectByStartAndStopIndices(files, startidx, stopidx)
        return files

    def read(self, datapath, ext=None, startidx=None, stopidx=None, recursive=False):
        """Sets up Spark RDD across files specified by datapath on local filesystem.

        Returns RDD of <string filepath, string buffer> k/v pairs.
        """
        abspath = self.uriToPath(datapath)
        filepaths = self.listFiles(abspath, ext=ext, startidx=startidx, stopidx=stopidx, recursive=recursive)

        lfilepaths = len(filepaths)
        self.lastnrecs = lfilepaths
        return self.sc.parallelize(enumerate(filepaths), lfilepaths).map(lambda (k, v): (k, _localRead(v)))


class _BotoS3Client(object):
    """Superclass for boto-based S3 readers.
    """
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
        """Initialization; validates that AWS keys are available as environment variables.

        Will let boto library look up credentials itself according to its own rules - e.g. first looking for
        AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, then going through several possible config files and finally
        looking for a ~/.aws/credentials .ini-formatted file. See boto docs:
        http://boto.readthedocs.org/en/latest/boto_config_tut.html
        """
        if not _have_boto:
            raise ValueError("The boto package does not appear to be available; boto is required for BotoS3Reader")


class BotoS3ParallelReader(_BotoS3Client):
    """Parallel reader backed by boto AWS client library.
    """
    def __init__(self, sparkcontext):
        super(BotoS3ParallelReader, self).__init__()
        self.sc = sparkcontext
        self.lastnrecs = None

    def listFiles(self, datapath, ext=None, startidx=None, stopidx=None, recursive=False):
        bucketname, keynames = self._listFilesImpl(datapath, ext=ext, startidx=startidx, stopidx=stopidx,
                                                   recursive=recursive)
        return ["s3n:///%s/%s" % (bucketname, keyname) for keyname in keynames]

    def _listFilesImpl(self, datapath, ext=None, startidx=None, stopidx=None, recursive=False):
        if recursive:
            raise NotImplementedError("Recursive traversal of directories isn't yet implemented for S3 - sorry!")
        parse = _BotoS3Client.parseS3Query(datapath)

        conn = boto.connect_s3()
        bucket = conn.get_bucket(parse[0])
        keys = _BotoS3Client.retrieveKeys(bucket, parse[1], prefix=parse[2], postfix=parse[3])
        keynamelist = [key.name for key in keys]
        if ext:
            keynamelist = [keyname for keyname in keynamelist if keyname.endswith(ext)]
        keynamelist.sort()

        keynamelist = selectByStartAndStopIndices(keynamelist, startidx, stopidx)
        return bucket.name, keynamelist

    def read(self, datapath, ext=None, startidx=None, stopidx=None, recursive=False):
        """Sets up Spark RDD across S3 objects specified by datapath.

        Returns RDD of <string s3 keyname, string buffer> k/v pairs.
        """
        if recursive:
            raise NotImplementedError("Recursive traversal of directories isn't yet implemented for S3 - sorry!")
        datapath = appendExtensionToPathSpec(datapath, ext)
        bucketname, keynamelist = self._listFilesImpl(datapath, startidx=startidx, stopidx=stopidx)

        if not keynamelist:
            raise FileNotFoundError("No S3 objects found for '%s'" % datapath)

        # originally this was set up so as to try to conserve boto connections, with one boto connection
        # d/ling multiple images - hence this being structured as a mapPartitions() function rather than
        # a straight map. but it is probably better to maintain one partition per image. all of which is
        # to say, there is no longer any good reason for this to be a mapPartitions() call rather than
        # just a map(), since now the parallelize call below is setting number partitions equal to number
        # images, but it works so I'm not yet messing with it.
        def readSplitFromS3(kvIter):
            conn = boto.connect_s3()
            bucket = conn.get_bucket(bucketname)
            for kv in kvIter:
                idx, keyname = kv
                key = bucket.get_key(keyname)
                buf = key.get_contents_as_string()
                yield idx, buf

        # don't specify number of splits here - allow reuse of connections within partition
        self.lastnrecs = len(keynamelist)
        # now set num partitions explicitly to num images - see comment on readSplitFromS3 above.
        return self.sc.parallelize(enumerate(keynamelist), self.lastnrecs).mapPartitions(readSplitFromS3)


class LocalFSFileReader(object):
    """File reader backed by python's native file() objects.
    """
    def __listRecursive(self, datapath):
        if os.path.isdir(datapath):
            dirname = datapath
            matchpattern = None
        else:
            dirname, matchpattern = os.path.split(datapath)

        filenames = set()
        for root, dirs, files in os.walk(dirname):
            if matchpattern:
                files = fnmatch.filter(files, matchpattern)
            for filename in files:
                filenames.add(os.path.join(root, filename))
        filenames = list(filenames)
        filenames.sort()
        return filenames

    def list(self, datapath, filename=None, startidx=None, stopidx=None, recursive=False):
        """List files specified by datapath.

        Datapath may include a single wildcard ('*') in the filename specifier.

        Returns sorted list of absolute path strings.
        """
        abspath = LocalFSParallelReader.uriToPath(datapath)

        if (not filename) and recursive:
            return self.__listRecursive(abspath)

        if filename:
            if os.path.isdir(abspath):
                abspath = os.path.join(abspath, filename)
            else:
                abspath = os.path.join(os.path.dirname(abspath), filename)
        else:
            if os.path.isdir(abspath):
                abspath = os.path.join(abspath, "*")

        files = glob.glob(abspath)
        # filter out directories
        files = [fpath for fpath in files if not os.path.isdir(fpath)]
        files.sort()
        files = selectByStartAndStopIndices(files, startidx, stopidx)
        return files

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
    """File reader backed by the boto AWS client library.
    """
    def __getMatchingKeys(self, datapath, filename=None):
        parse = _BotoS3Client.parseS3Query(datapath)
        conn = boto.connect_s3()
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

        return _BotoS3Client.retrieveKeys(bucket, keyname, prefix=parse[2], postfix=parse[3])

    def list(self, datapath, filename=None, startidx=None, stopidx=None, recursive=False):
        """List s3 objects specified by datapath.

        Returns sorted list of 's3n://' URIs.
        """
        if recursive:
            raise NotImplementedError("Recursive traversal of directories isn't yet implemented for S3 - sorry!")

        keys = self.__getMatchingKeys(datapath, filename=filename)
        keynames = ["s3n:///" + key.bucket.name + "/" + key.name for key in keys]
        keynames.sort()
        keynames = selectByStartAndStopIndices(keynames, startidx, stopidx)
        return keynames

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
    """Read-only file handle-like object exposing a subset of file methods.

    Returned by BotoS3FileReader's open() method.
    """
    def __init__(self, key):
        self._key = key
        self._closed = False
        self._offset = 0

    def close(self):
        try:
            self._key.close(fast=True)
        except TypeError:
            # workaround for early versions of boto that don't have the 'fast' keyword
            self._key.close()
        self._closed = True

    def read(self, size=-1):
        if self._offset or (size > -1):
            # return empty string to indicate EOF if we are offset past the end of the file
            # else boto will throw an error at us
            if self._offset >= self._key.size:
                return ""
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
        return "s3n:///" + self._key.bucket.name + "/" + self._key.name

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
    """Helper function used by get*ForPath().
    """
    parseresult = urlparse.urlparse(datapath)
    clazz = lookup.get(parseresult.scheme, default)
    if clazz is None:
        raise NotImplementedError("No implementation for scheme " + parseresult.scheme)
    return clazz


def getParallelReaderForPath(datapath):
    """Returns the class of a parallel reader suitable for the scheme used by `datapath`.

    The resulting class object must still be instantiated in order to get a usable instance of the class.

    Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
    Returns LocalFSParallelReader if scheme is absent or not recognized.
    """
    return getByScheme(datapath, SCHEMAS_TO_PARALLELREADERS, LocalFSParallelReader)


def getFileReaderForPath(datapath):
    """Returns the class of a file reader suitable for the scheme used by `datapath`.

    The resulting class object must still be instantiated in order to get a usable instance of the class.

    Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
    Returns LocalFSFileReader if scheme is absent or not recognized.
    """
    return getByScheme(datapath, SCHEMAS_TO_FILEREADERS, LocalFSFileReader)
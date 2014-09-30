import os
import shutil
import urllib
import urlparse

_have_boto = False
try:
    import boto
    _have_boto = True
except ImportError:
    pass


class LocalFSParallelWriter(object):
    def __init__(self, datapath, overwrite=False):
        self._datapath = datapath
        # thanks stack overflow:
        # http://stackoverflow.com/questions/5977576/is-there-a-convenient-way-to-map-a-file-uri-to-os-path
        self._abspath = urllib.url2pathname(urlparse.urlparse(datapath).path)
        self._overwrite = overwrite
        self._checked = False
        self._checkDirectory()

    def _checkDirectory(self):
        if not self._checked:
            if os.path.isfile(self._abspath):
                raise ValueError("LocalFSParallelWriter must be initialized with path to directory not file" +
                                 " in order to use writerFcn. Got: " + self._datapath)
            if os.path.isdir(self._abspath):
                if self._overwrite:
                    shutil.rmtree(self._abspath)
                else:
                    raise ValueError("Directory %s already exists, and overwrite is false" % self._datapath)
            os.mkdir(self._abspath)
            self._checked = True

    def writerFcn(self, kv):
        label, buf = kv
        with open(os.path.join(self._abspath, label), 'wb') as f:
            f.write(buf)


class _BotoS3Writer(object):
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


class BotoS3ParallelWriter(_BotoS3Writer):
    def __init__(self, datapath, overwrite=False):
        super(BotoS3ParallelWriter, self).__init__()
        self._datapath = datapath
        self._overwrite = overwrite
        self._contextactive = False
        self._conn = None
        self._keyname = None
        self._bucket = None

    def _setupContext(self):
        """
        Set up a boto s3 connection.

        This is expected to end up being called once for each spark worker.
        """
        conn = boto.connect_s3(self._access_key, self._secret_key)
        bucketname, keyname = _BotoS3Writer._parseS3Schema(self._datapath)
        if not keyname.endswith("/"):
            keyname += "/"
        bucket = conn.get_bucket(bucketname)

        self._conn = conn
        self._keyname = keyname
        self._bucket = bucket
        self._contextactive = True

    def writerFcn(self, kv):
        if not self._contextactive:
            self._setupContext()

        label, buf = kv
        s3key = boto.s3.key.Key(self._bucket)
        s3key.name = self._keyname + label
        s3key.set_contents_from_string(buf)


class LocalFSFileWriter(object):
    def __init__(self, datapath, filename, overwrite=False):
        self._datapath = datapath
        self._filename = filename
        self._abspath = os.path.join(urllib.url2pathname(urlparse.urlparse(datapath).path), filename)
        self._overwrite = overwrite
        self._checked = False

    def _checkWriteFile(self):
        if not self._checked:
            if os.path.isdir(self._abspath):
                raise ValueError("LocalFSFileWriter must be initialized with path to file, not directory," +
                                 " in order to use writeFile. Got path: '%s', filename: '%s'" %
                                 (self._datapath, self._filename))
            if (not self._overwrite) and os.path.exists(self._abspath):
                raise ValueError("File %s already exists, and overwrite is false" % self._datapath)
            self._checked = True

    def writeFile(self, buf):
        self._checkWriteFile()
        with open(os.path.join(self._abspath), 'wb') as f:
            f.write(buf)


class LocalFSCollectedFileWriter(object):
    def __init__(self, datapath, overwrite=False):
        self._datapath = datapath
        self._abspath = urllib.url2pathname(urlparse.urlparse(datapath).path)
        self._overwrite = overwrite
        self._checked = False

    def _checkDirectory(self):
        # todo: this is duplicated code with LocalFSParallelWriter
        if not self._checked:
            if os.path.isfile(self._abspath):
                raise ValueError("LocalFSCollectedFileWriter must be initialized with path to directory not file" +
                                 " in order to use writerFcn. Got: " + self._datapath)
            if os.path.isdir(self._abspath):
                if self._overwrite:
                    shutil.rmtree(self._abspath)
                else:
                    raise ValueError("Directory %s already exists, and overwrite is false" % self._datapath)
            os.mkdir(self._abspath)  # will throw error if is already a file
            self._checked = True

    def writeCollectedFiles(self, labelBufSequence):
        self._checkDirectory()
        for filename, buf in labelBufSequence:
            abspath = os.path.join(self._abspath, filename)
            with open(abspath, 'wb') as f:
                f.write(buf)

SCHEMAS_TO_PARALLELWRITERS = {
    '': LocalFSParallelWriter,
    'file': LocalFSParallelWriter,
    's3': None,
    's3n': None,
    'hdfs': None
}

SCHEMAS_TO_FILEWRITERS = {
    '': LocalFSFileWriter,
    'file': LocalFSFileWriter,
    's3': None,
    's3n': None,
    'hdfs': None
}

SCHEMAS_TO_COLLECTEDFILEWRITERS = {

    '': LocalFSCollectedFileWriter,
    'file': LocalFSCollectedFileWriter,
    's3': None,
    's3n': None,
    'hdfs': None
}

def __getWriter(datapath, lookup, default):
    parseresult = urlparse.urlparse(datapath)
    clazz = lookup.get(parseresult.scheme, default)
    if clazz is None:
        raise NotImplementedError("No writer implemented for scheme " + parseresult.scheme)
    return clazz


def getParallelWriterForPath(datapath):
    return __getWriter(datapath, SCHEMAS_TO_PARALLELWRITERS, LocalFSParallelWriter)


def getFileWriterForPath(datapath):
    return __getWriter(datapath, SCHEMAS_TO_FILEWRITERS, LocalFSFileWriter)

def getCollectedFileWriterForPath(datapath):
    return __getWriter(datapath, SCHEMAS_TO_COLLECTEDFILEWRITERS, LocalFSCollectedFileWriter)
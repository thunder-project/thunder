"""Classes that abstract writing to various types of filesystems.

Currently two types of 'filesystem' are supported:

* the local file system, via python's native file() objects

* Amazon's S3, using the boto library (only if boto is installed; boto is not a requirement)

For each filesystem, three types of writer classes are provided:

* parallel writers are intended to serve as a data sink at the end of a Spark workflow. They provide a `writerFcn(kv)`
method, which is intended to be used inside a Spark foreach() call (for instance: myrdd.foreach(writer.writerFcn)).
They expect to be given key, value pairs where the key is a filename (not including a directory component in the path),
and the value is a string buffer.

* file writers abstract across the supported filesystems, providing a common writeFile(buf) interface that writes
the contents of buf to a file object specified at writer initialization.

* collected file writers are intended to be used after a Spark collect() call. They provide similar functionality
to the parallel writers, but operate only on the driver rather than distributed across all nodes. This is intended
to make it easier to write the results of an analysis to the local filesystem, without requiring NFS or a similar
distributed file system available on all worker nodes.

"""
import os
import shutil
import urllib
import urlparse

from thunder.rdds.fileio.readers import _BotoS3Client, getByScheme


_have_boto = False
try:
    import boto
    _have_boto = True
except ImportError:
    boto = None


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


class _BotoS3Writer(_BotoS3Client):
    def __init__(self):
        super(_BotoS3Writer, self).__init__()

        self._contextactive = False
        self._conn = None
        self._keyname = None
        self._bucket = None

    def activateContext(self, datapath, isDirectory):
        """
        Set up a boto s3 connection.

        This is expected to end up being called once for each spark worker.
        """
        conn = boto.connect_s3(self.accessKey, self.secretKey)
        parsed = _BotoS3Client.parseS3Query(datapath)
        bucketname = parsed[0]
        keyname = parsed[1]
        if isDirectory and (not keyname.endswith("/")):
            keyname += "/"
        bucket = conn.get_bucket(bucketname)

        self._conn = conn
        self._keyname = keyname
        self._bucket = bucket
        self._contextactive = True

    @property
    def bucket(self):
        return self._bucket

    @property
    def keyname(self):
        return self._keyname

    @property
    def contextActive(self):
        return self._contextactive


class BotoS3ParallelWriter(_BotoS3Writer):
    # todo: needs to check before writing if overwrite is True
    def __init__(self, datapath, overwrite=False):
        super(BotoS3ParallelWriter, self).__init__()
        self._datapath = datapath
        self._overwrite = overwrite

    def writerFcn(self, kv):
        if not self.contextActive:
            self.activateContext(self._datapath, True)

        label, buf = kv
        s3key = boto.s3.key.Key(self.bucket)
        s3key.name = self.keyname + label
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


class BotoS3FileWriter(_BotoS3Writer):
    # todo: needs to check before writing if overwrite is True
    def __init__(self, datapath, filename, overwrite=False):
        super(BotoS3FileWriter, self).__init__()
        self._datapath = datapath
        self._filename = filename
        self._overwrite = overwrite

    def writeFile(self, buf):
        if not self.contextActive:
            self.activateContext(self._datapath, True)

        s3key = boto.s3.key.Key(self.bucket)
        s3key.name = self.keyname + self._filename
        s3key.set_contents_from_string(buf)


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


class BotoS3CollectedFileWriter(_BotoS3Writer):
    # todo: needs to check before writing if overwrite is True
    def __init__(self, datapath, overwrite=False):
        super(BotoS3CollectedFileWriter, self).__init__()
        self._datapath = datapath
        self._overwrite = overwrite

    def writeCollectedFiles(self, labelBufSequence):
        if not self.contextActive:
            self.activateContext(self._datapath, True)

        for filename, buf in labelBufSequence:
            s3key = boto.s3.key.Key(self.bucket)
            s3key.name = self.keyname + filename
            s3key.set_contents_from_string(buf)


SCHEMAS_TO_PARALLELWRITERS = {
    '': LocalFSParallelWriter,
    'file': LocalFSParallelWriter,
    's3': BotoS3ParallelWriter,
    's3n': BotoS3ParallelWriter,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}

SCHEMAS_TO_FILEWRITERS = {
    '': LocalFSFileWriter,
    'file': LocalFSFileWriter,
    's3': BotoS3FileWriter,
    's3n': BotoS3FileWriter,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}

SCHEMAS_TO_COLLECTEDFILEWRITERS = {

    '': LocalFSCollectedFileWriter,
    'file': LocalFSCollectedFileWriter,
    's3': BotoS3CollectedFileWriter,
    's3n': BotoS3CollectedFileWriter,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}


def getParallelWriterForPath(datapath):
    """Returns the class of a parallel file writer suitable for the scheme used by `datapath`.

    The resulting class object must still be instantiated in order to get a usable instance of the class.

    Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
    Returns LocalFSParallelWriter if scheme is absent or not recognized.
    """
    return getByScheme(datapath, SCHEMAS_TO_PARALLELWRITERS, LocalFSParallelWriter)


def getFileWriterForPath(datapath):
    """Returns the class of a file writer suitable for the scheme used by `datapath`.

    The resulting class object must still be instantiated in order to get a usable instance of the class.

    Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
    Returns LocalFSFileWriter if scheme is absent or not recognized.
    """
    return getByScheme(datapath, SCHEMAS_TO_FILEWRITERS, LocalFSFileWriter)


def getCollectedFileWriterForPath(datapath):
    """Returns the class of a collected file writer suitable for the scheme used by `datapath`.

    The resulting class object must still be instantiated in order to get a usable instance of the class.

    Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
    Returns LocalFSCollectedFileWriter if scheme is absent or not recognized.
    """
    return getByScheme(datapath, SCHEMAS_TO_COLLECTEDFILEWRITERS, LocalFSCollectedFileWriter)
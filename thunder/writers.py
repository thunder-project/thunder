import os
import shutil
# standard library reorganization between Python 2 and 3
try:
    from urllib.parse import urlparse
    from urllib.request import url2pathname
except ImportError:
    from urlparse import urlparse
    from urllib import url2pathname

from thunder.readers import BotoClient, get_by_scheme


class LocalParallelWriter(object):
    def __init__(self, path, overwrite=False, **kwargs):
        self._path = url2pathname(urlparse(path).path)
        self._overwrite = overwrite
        self._checked = False
        self.check_directory()

    def check_directory(self):
        if not self._checked:
            if os.path.isfile(self._path):
                raise ValueError("LocalFSParallelWriter must be initialized with path "
                                 "to directory not file in order to use writer. Got: " + self._path)
            if os.path.isdir(self._path):
                if self._overwrite:
                    shutil.rmtree(self._path)
                else:
                    raise ValueError("Directory %s already exists, "
                                     "and overwrite is false" % self._path)
            os.mkdir(self._path)
            self._checked = True

    def write(self, kv):
        label, buf = kv
        with open(os.path.join(self._path, label), 'wb') as f:
            f.write(buf)


class BotoWriter(BotoClient):
    def __init__(self, credentials=None):
        super(BotoWriter, self).__init__(credentials=credentials)
        self._scheme = None
        self._active = False
        self._conn = None
        self._key = None
        self._bucket = None

    def activate(self, path, isdirectory):
        """
        Set up a boto connection.
        """
        from .utils import connection_with_anon, connection_with_gs

        parsed = BotoClient.parse_query(path)

        scheme = parsed[0]
        bucket_name = parsed[1]
        key = parsed[2]

        if scheme == 's3' or scheme == 's3n':
            conn = connection_with_anon(self.credentials)
            bucket = conn.get_bucket(bucket_name)
        elif scheme == 'gs':
            conn = connection_with_gs(bucket_name)
            bucket = conn.get_bucket()
        else:
            raise NotImplementedError("No file reader implementation for URL scheme " + scheme)

        if isdirectory and (not key.endswith("/")):
            key += "/"

        self._scheme = scheme
        self._conn = conn
        self._key = key
        self._bucket = bucket
        self._active = True

    @property
    def bucket(self):
        return self._bucket

    @property
    def key(self):
        return self._key

    @property
    def active(self):
        return self._active


class BotoParallelWriter(BotoWriter):
    def __init__(self, path, overwrite=False, credentials=None):
        super(BotoParallelWriter, self).__init__(credentials=credentials)
        self._path = path
        self._overwrite = overwrite

    def write(self, kv):
        if not self.active:
            self.activate(self._path, True)
        label, buf = kv
        self._bucket.new_key(self.key + label).set_contents_from_string(buf)


class LocalFileWriter(object):
    def __init__(self, path, filename, overwrite=False, **kwargs):
        self._path = os.path.join(url2pathname(urlparse(path).path), filename)
        self._filename = filename
        self._overwrite = overwrite
        self._checked = False

    def check_file(self):
        if not self._checked:
            if os.path.isdir(self._path):
                raise ValueError("LocalFileWriter must be initialized with path to file, "
                                 "not directory, in order to use writeFile. "
                                 "Got path: '%s', filename: '%s'" % (self._path, self._filename))
            if (not self._overwrite) and os.path.exists(self._path):
                raise ValueError("File %s already exists, and overwrite is false" % self._path)
            self._checked = True

    def write(self, buf):
        self.check_file()
        with open(os.path.join(self._path), 'wb') as f:
            f.write(buf.encode('utf-8'))


class BotoFileWriter(BotoWriter):
    def __init__(self, path, filename, overwrite=False, credentials=None):
        super(BotoFileWriter, self).__init__(credentials=credentials)
        self._path = path
        self._filename = filename
        self._overwrite = overwrite

    def write(self, buf):
        if not self.active:
            self.activate(self._path, True)
        self._bucket.new_key(self.key + self._filename).set_contents_from_string(buf)


SCHEMAS_TO_PARALLELWRITERS = {
    '': LocalParallelWriter,
    'file': LocalParallelWriter,
    'gs': BotoParallelWriter,
    's3': BotoParallelWriter,
    's3n': BotoParallelWriter,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}

SCHEMAS_TO_FILEWRITERS = {
    '': LocalFileWriter,
    'file': LocalFileWriter,
    'gs': BotoFileWriter,
    's3': BotoFileWriter,
    's3n': BotoFileWriter,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}

def get_parallel_writer(path):
    """
    Returns the class of a parallel file writer for the scheme in path.

    The resulting class object must still be instantiated.
    Throws NotImplementedError if the requested scheme is not supported (e.g. "ftp://").
    Returns LocalFileReader if scheme is absent or not recognized.
    """
    return get_by_scheme(path, SCHEMAS_TO_PARALLELWRITERS, LocalParallelWriter)

def get_file_writer(path):
    """
    Returns the class of a file writer suitable for the scheme in path.

    The resulting class object must still be instantiated.
    Throws NotImplementedError if the requested scheme is not supported (e.g. "ftp://").
    Returns LocalFileReader if scheme is absent or not recognized.
    """
    return get_by_scheme(path, SCHEMAS_TO_FILEWRITERS, LocalFileWriter)
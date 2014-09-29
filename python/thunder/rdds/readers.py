import glob
import os
import urlparse

_have_boto = False
try:
    import boto
    _have_boto = True
except ImportError:
    pass


class LocalFSReader(object):
    def __init__(self, sparkcontext):
        self.sc = sparkcontext
        self.lastnrecs = None

    @staticmethod
    def listFiles(datapath, ext=None, startidx=None, stopidx=None):

        if os.path.isdir(datapath):
            if ext:
                files = sorted(glob.glob(os.path.join(datapath, '*.' + ext)))
            else:
                files = sorted(os.listdir(datapath))
        else:
            files = sorted(glob.glob(datapath))

        if len(files) < 1:
            raise IOError('cannot find files of type "%s" in %s' % (ext if ext else '*', datapath))

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
        filepaths = self.listFiles(datapath, ext=ext, startidx=startidx, stopidx=stopidx)

        def readfcn(filepath):
            buf = None
            with open(filepath, 'rb') as f:
                buf = f.read()
            return buf

        lfilepaths = len(filepaths)
        self.lastnrecs = lfilepaths
        return self.sc.parallelize(enumerate(filepaths), lfilepaths).map(lambda (k, v): (k, readfcn(v)))


class BotoS3Reader(object):
    @staticmethod
    def _parseS3Schema(datapath):
        parseresult = urlparse.urlparse(datapath)
        return parseresult.netloc, parseresult.path.lstrip("/")

    def __init__(self, sparkcontext):
        if not _have_boto:
            raise ValueError("The boto package does not appear to be available; boto is required for BotoS3Reader")
        if (not 'AWS_ACCESS_KEY_ID' in os.environ) or (not 'AWS_SECRET_ACCESS_KEY' in os.environ):
            raise ValueError("The environment variables 'AWS_ACCESS_KEY_ID' and 'AWS_SECRET_ACCESS_KEY' must be set in order to read from s3")

        # save keys in this object and serialize out to workers to prevent having to set env vars separately on all
        # nodes in the cluster
        self._access_key = os.environ['AWS_ACCESS_KEY_ID']
        self._secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

        self.sc = sparkcontext
        self.lastnrecs = None

    def _listFiles(self, datapath, ext=None, startidx=None, stopidx=None):
        bucketname, keyname = BotoS3Reader._parseS3Schema(datapath)
        conn = boto.connect_s3(aws_access_key_id=self._access_key, aws_secret_access_key=self._secret_key)
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

        access_key = self._access_key
        secret_key = self._secret_key

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


SCHEMAS_TO_READERS = {
    '': LocalFSReader,
    'file': LocalFSReader,
    's3': BotoS3Reader,
    's3n': BotoS3Reader
}


def getReaderForPath(datapath):
    parseresult = urlparse.urlparse(datapath)
    return SCHEMAS_TO_READERS.get(parseresult.scheme, LocalFSReader)
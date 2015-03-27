"""Classes that abstract writing to various types of filesystems.

Currently two types of 'filesystem' are supported:

* the local file system, via python's native file() objects

* Amazon's S3 or Google Storage, using the boto library (only if boto is installed; boto is not a requirement)

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

from thunder.rdds.fileio.readers import _BotoClient, getByScheme


_haveBoto = False
try:
	import boto
	_haveBoto = True
except ImportError:
	boto = None


class LocalFSParallelWriter(object):
	def __init__(self, dataPath, overwrite=False, **kwargs):
		self._dataPath = dataPath
		# thanks stack overflow:
		# http://stackoverflow.com/questions/5977576/is-there-a-convenient-way-to-map-a-file-uri-to-os-path
		self._absPath = urllib.url2pathname(urlparse.urlparse(dataPath).path)
		self._overwrite = overwrite
		self._checked = False
		self._checkDirectory()

	def _checkDirectory(self):
		if not self._checked:
			if os.path.isfile(self._absPath):
				raise ValueError("LocalFSParallelWriter must be initialized with path to directory not file" +
								 " in order to use writerFcn. Got: " + self._dataPath)
			if os.path.isdir(self._absPath):
				if self._overwrite:
					shutil.rmtree(self._absPath)
				else:
					raise ValueError("Directory %s already exists, and overwrite is false" % self._dataPath)
			os.mkdir(self._absPath)
			self._checked = True

	def writerFcn(self, kv):
		label, buf = kv
		with open(os.path.join(self._absPath, label), 'wb') as f:
			f.write(buf)


class _BotoWriter(_BotoClient):
	def __init__(self, awsCredentialsOverride=None):
		super(_BotoWriter, self).__init__(awsCredentialsOverride=awsCredentialsOverride)
		self._storageScheme = None
		self._contextActive = False
		self._conn = None
		self._keyName = None
		self._bucket = None

	def activateContext(self, dataPath, isDirectory):
		"""
		Set up a boto connection.

		"""
		parsed = _BotoClient.parseQuery(dataPath)

		storageScheme = parsed[0]
		if storageScheme == 's3' or storageScheme == 's3n':
			from boto.s3.connection import S3Connection
			conn = S3Connection(**self.awsCredentialsOverride.credentialsAsDict)
		elif storageScheme == 'gs':
			from boto.GS.connection import GSConnection
			conn = GSConnection()
		else:
			raise NotImplementedError("No file reader implementation for URL scheme " + storageScheme)

		bucketName = parsed[1]
		keyName = parsed[2]
		if isDirectory and (not keyName.endswith("/")):
			keyName += "/"
		bucket = conn.get_bucket(bucketName)

		self._storageScheme = storageScheme
		self._conn = conn
		self._keyName = keyName
		self._bucket = bucket
		self._contextActive = True

	@property
	def bucket(self):
		return self._bucket

	@property
	def keyName(self):
		return self._keyName

	@property
	def contextActive(self):
		return self._contextActive


class BotoParallelWriter(_BotoWriter):
	def __init__(self, dataPath, overwrite=False, awsCredentialsOverride=None):
		super(BotoParallelWriter, self).__init__(awsCredentialsOverride=awsCredentialsOverride)
		self._dataPath = dataPath
		self._overwrite = overwrite

	def writerFcn(self, kv):
		if not self.contextActive:
			self.activateContext(self._dataPath, True)
		label, buf = kv
		self._bucket.new_key(self.keyName + label).set_contents_from_string(buf)

class LocalFSFileWriter(object):
	def __init__(self, dataPath, filename, overwrite=False, **kwargs):
		self._dataPath = dataPath
		self._filename = filename
		self._absPath = os.path.join(urllib.url2pathname(urlparse.urlparse(dataPath).path), filename)
		self._overwrite = overwrite
		self._checked = False

	def _checkWriteFile(self):
		if not self._checked:
			if os.path.isdir(self._absPath):
				raise ValueError("LocalFSFileWriter must be initialized with path to file, not directory," +
								 " in order to use writeFile. Got path: '%s', filename: '%s'" %
								 (self._dataPath, self._filename))
			if (not self._overwrite) and os.path.exists(self._absPath):
				raise ValueError("File %s already exists, and overwrite is false" % self._dataPath)
			self._checked = True

	def writeFile(self, buf):
		self._checkWriteFile()
		with open(os.path.join(self._absPath), 'wb') as f:
			f.write(buf)


class BotoFileWriter(_BotoWriter):
	def __init__(self, dataPath, filename, overwrite=False, awsCredentialsOverride=None):
		super(BotoFileWriter, self).__init__(awsCredentialsOverride=awsCredentialsOverride)
		self._dataPath = dataPath
		self._filename = filename
		self._overwrite = overwrite

	def writeFile(self, buf):
		if not self.contextActive:
			self.activateContext(self._dataPath, True)
		self._bucket.new_key(self.keyName + self._filename).set_contents_from_string(buf)

class LocalFSCollectedFileWriter(object):
	def __init__(self, dataPath, overwrite=False, **kwargs):
		self._dataPath = dataPath
		self._absPath = urllib.url2pathname(urlparse.urlparse(dataPath).path)
		self._overwrite = overwrite
		self._checked = False

	def _checkDirectory(self):
		# todo: this is duplicated code with LocalFSParallelWriter
		if not self._checked:
			if os.path.isfile(self._absPath):
				raise ValueError("LocalFSCollectedFileWriter must be initialized with path to directory not file" +
								 " in order to use writerFcn. Got: " + self._dataPath)
			if os.path.isdir(self._absPath):
				if self._overwrite:
					shutil.rmtree(self._absPath)
				else:
					raise ValueError("Directory %s already exists, and overwrite is false" % self._dataPath)
			os.mkdir(self._absPath)  # will throw error if is already a file
			self._checked = True

	def writeCollectedFiles(self, labelBufSequence):
		self._checkDirectory()
		for filename, buf in labelBufSequence:
			absPath = os.path.join(self._absPath, filename)
			with open(absPath, 'wb') as f:
				f.write(buf)


class BotoCollectedFileWriter(_BotoWriter):
	# todo: needs to check before writing if overwrite is True
	def __init__(self, dataPath, overwrite=False, awsCredentialsOverride=None):
		super(BotoCollectedFileWriter, self).__init__(awsCredentialsOverride=awsCredentialsOverride)
		self._dataPath = dataPath
		self._overwrite = overwrite

	def writeCollectedFiles(self, labelBufSequence):
		if not self.contextActive:
			self.activateContext(self._dataPath, True)

		for filename, buf in labelBufSequence:
			self._bucket.new_key(self.keyName + filename).set_contents_from_string(buf)


SCHEMAS_TO_PARALLELWRITERS = {
	'': LocalFSParallelWriter,
	'file': LocalFSParallelWriter,
	'gs': BotoParallelWriter,
	's3': BotoParallelWriter,
	's3n': BotoParallelWriter,
	'hdfs': None,
	'http': None,
	'https': None,
	'ftp': None
}

SCHEMAS_TO_FILEWRITERS = {
	'': LocalFSFileWriter,
	'file': LocalFSFileWriter,
	'gs': BotoFileWriter,
	's3': BotoFileWriter,
	's3n': BotoFileWriter,
	'hdfs': None,
	'http': None,
	'https': None,
	'ftp': None
}

SCHEMAS_TO_COLLECTEDFILEWRITERS = {
	'': LocalFSCollectedFileWriter,
	'file': LocalFSCollectedFileWriter,
	'gs': BotoCollectedFileWriter,
	's3': BotoCollectedFileWriter,
	's3n': BotoCollectedFileWriter,
	'hdfs': None,
	'http': None,
	'https': None,
	'ftp': None
}


def getParallelWriterForPath(dataPath):
	"""Returns the class of a parallel file writer suitable for the scheme used by `datapath`.

	The resulting class object must still be instantiated in order to get a usable instance of the class.

	Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
	Returns LocalFSParallelWriter if scheme is absent or not recognized.
	"""
	return getByScheme(dataPath, SCHEMAS_TO_PARALLELWRITERS, LocalFSParallelWriter)


def getFileWriterForPath(dataPath):
	"""Returns the class of a file writer suitable for the scheme used by `datapath`.

	The resulting class object must still be instantiated in order to get a usable instance of the class.

	Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
	Returns LocalFSFileWriter if scheme is absent or not recognized.
	"""
	return getByScheme(dataPath, SCHEMAS_TO_FILEWRITERS, LocalFSFileWriter)


def getCollectedFileWriterForPath(dataPath):
	"""Returns the class of a collected file writer suitable for the scheme used by `datapath`.

	The resulting class object must still be instantiated in order to get a usable instance of the class.

	Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
	Returns LocalFSCollectedFileWriter if scheme is absent or not recognized.
	"""
	return getByScheme(dataPath, SCHEMAS_TO_COLLECTEDFILEWRITERS, LocalFSCollectedFileWriter)

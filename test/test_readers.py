import os
import shutil
import tempfile
import unittest
from nose.tools import assert_equal, assert_raises

from thunder.rdds.fileio.readers import LocalFSFileReader, LocalFSParallelReader, FileNotFoundError


def touch_empty(filepath):
    if not os.path.exists(filepath):
        dirname, filename = os.path.split(filepath)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        fp = open(filepath, 'w')
        fp.close()
    else:
        os.utime(filepath, None)


class TestCaseWithOutputDir(unittest.TestCase):
    def setUp(self):
        super(TestCaseWithOutputDir, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(TestCaseWithOutputDir, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestLocalReader(unittest.TestCase):

    def test_readRaisesFileNotFoundError(self):
        reader = LocalFSFileReader()
        assert_raises(FileNotFoundError, reader.read, "this directory doesnt exist", "definitely not with this file")


class TestLocalFileListing(TestCaseWithOutputDir):

    def _setup_files(self, filenames, expected=None):
        if expected:
            basenames = expected
        else:
            basenames = filenames
        expected = [os.path.join(self.outputdir, fname) for fname in basenames]
        del basenames

        inputFilepaths = [os.path.join(self.outputdir, fname) for fname in filenames]
        for fname in inputFilepaths:
            touch_empty(fname)
        return expected

    def _run_parallelReader_tst(self, filenames, recursive=False, expected=None):
        expected = self._setup_files(filenames, expected=expected)

        reader = LocalFSParallelReader(None)
        actual = reader.listFiles(self.outputdir, recursive=recursive)
        assert_equal(sorted(expected), actual)

    def _run_localReader_tst(self,  filenames, recursive=False, expected=None):
        expected = self._setup_files(filenames, expected=expected)
        reader = LocalFSFileReader()
        actual = reader.list(self.outputdir, recursive=recursive)
        assert_equal(sorted(expected), actual)

    def test_flatDirParallelRecursive(self):
        filenames = ["b", "a", "c"]
        self._run_parallelReader_tst(filenames, True)

    def test_flatDirLocalRecursive(self):
        filenames = ["b", "a", "c"]
        self._run_localReader_tst(filenames, True)

    def test_flatDirParallel(self):
        filenames = ["b", "a", "c"]
        self._run_parallelReader_tst(filenames, False)

    def test_flatDirLocal(self):
        filenames = ["b", "a", "c"]
        self._run_localReader_tst(filenames, False)

    def test_nestedDirParallelRecursive(self):
        filenames = ["foo/b", "foo/bar/q", "bar/a", "c"]
        self._run_parallelReader_tst(filenames, True)

    def test_nestedDirLocalRecursive(self):
        filenames = ["foo/b", "foo/bar/q", "bar/a", "c"]
        self._run_localReader_tst(filenames, True)

    def test_nestedDirParallel(self):
        filenames = ["foo/b", "foo/bar/q", "bar/a", "c"]
        self._run_parallelReader_tst(filenames, False, expected=["c"])

    def test_nestedDirLocal(self):
        filenames = ["foo/b", "foo/bar/q", "bar/a", "c"]
        self._run_localReader_tst(filenames, False, expected=["c"])



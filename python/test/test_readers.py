import unittest
from nose.tools import assert_raises

from thunder.rdds.fileio.readers import LocalFSFileReader, FileNotFoundError


class TestLocalReader(unittest.TestCase):

    def test_readRaisesFileNotFoundError(self):
        reader = LocalFSFileReader()
        assert_raises(FileNotFoundError, reader.read, "this directory doesnt exist", "definitely not with this file")
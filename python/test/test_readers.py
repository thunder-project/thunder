from nose.tools import assert_equals, assert_true, assert_almost_equal, assert_raises
import unittest
from thunder.rdds.readers import LocalFSFileReader, FileNotFoundError


class TestLocalReader(unittest.TestCase):

    def test_readRaisesFileNotFoundError(self):
        reader = LocalFSFileReader()
        assert_raises(FileNotFoundError, reader.read, "this directory doesnt exist", "definitely not with this file")
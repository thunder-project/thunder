import shutil
import struct
import tempfile
import os
import unittest
import logging
from numpy import dtype, array, allclose
from nose.tools import assert_equals, assert_true
from pyspark import SparkContext, SparkConf
from thunder.rdds.series import SeriesLoader

class PySparkTestCase(unittest.TestCase):
    def setUp(self):
        class_name = self.__class__.__name__
        logging.getLogger("py4j").setLevel(logging.WARNING)
        #self.sc = SparkContext('local', class_name)
        #conf = SparkConf()
        self.sc = SparkContext('local', class_name)

    def tearDown(self):
        self.sc.stop()
        # To avoid Akka rebinding to the same port, since it doesn't unbind
        # immediately on shutdown
        self.sc._jvm.System.clearProperty("spark.driver.port")


class RDDsSparkTestCase(PySparkTestCase):
    def setUp(self):
        super(RDDsSparkTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(RDDsSparkTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class SeriesBinaryTestData(object):
    __slots__ = ('keys', 'vals', 'keyDType', 'valDType')

    def __init__(self, keys, vals, keydtype, valdtype):
        self.keys = keys
        self.vals = vals
        self.keyDType = keydtype
        self.valDType = valdtype

    @property
    def keyStructFormat(self):
        return self.keyDType.char * self.nkeys

    @property
    def valStructFormat(self):
        return self.valDType.char * self.nvals

    @property
    def data(self):
        return zip(self.keys, self.vals)

    @property
    def nkeys(self):
        return len(self.keys[0])

    @property
    def nvals(self):
        return len(self.vals[0])

    @classmethod
    def _validateLengths(cls, dat):
        l = len(dat[0])
        for d in dat:
            assert len(d) == l, "Data of unequal lengths, %d and %d" % (l, len(d))

    @classmethod
    def _normalizeDType(cls, dtypeinst, data):
        if dtypeinst is None:
            return data.dtype
        return dtype(dtypeinst)

    @classmethod
    def fromArrays(cls, keys, vals, keydtype=None, valdtype=None):
        # validation:
        keydtype = cls._normalizeDType(keydtype, keys)
        valdtype = cls._normalizeDType(valdtype, vals)

        assert len(keys) == len(vals), "Unequal numbers of keys and values, %d and %d" % (len(keys), len(vals))
        cls._validateLengths(keys)
        cls._validateLengths(vals)
        return cls(keys, vals, keydtype, valdtype)


class TestSeriesBinaryLoader(RDDsSparkTestCase):

    def test_fromBinary(self):
        # run this as a single big test so as to avoid repeated setUp and tearDown of the spark context
        DATA = []
        # data will be a sequence of test data
        # each test data item will be a sequence of key, value pairs
        # each key, value pair will be a 2-tuple of numpy arrays
        # all keys and all values in a test data item must be of the same length
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int16'))

        for itemidx, item in enumerate(DATA):
            fname = os.path.join(self.outputdir, 'inputfile%d.bin' % itemidx)
            with open(fname, 'wb') as f:
                for keys, vals in item.data:
                    f.write(struct.pack(item.keyStructFormat, *keys))
                    f.write(struct.pack(item.valStructFormat, *vals))

            loader = SeriesLoader(item.nkeys, item.nvals, keytype=str(item.keyDType), valuetype=str(item.valDType))
            series = loader.fromBinary(fname, self.sc)
            seriesdata = series.rdd.collect()

            expecteddata = item.data
            assert_equals(len(expecteddata), len(seriesdata),
                          "Differing numbers of k/v pairs in item %d; expected %d, got %d" %
                          (itemidx, len(expecteddata), len(seriesdata)))

            for expected, actual in zip(expecteddata, seriesdata):
                expectedkeys = tuple(expected[0])
                expectedvals = array(expected[1], dtype=item.valDType)
                assert_equals(expectedkeys, actual[0],
                              "Key mismatch in item %d; expected %s, got %s" %
                              (itemidx, str(expectedkeys), str(actual[0])))
                assert_true(allclose(expectedvals, actual[1]),
                            "Value mismatch in item %d; expected %s, got %s" %
                            (itemidx, str(expectedvals), str(actual[1])))



import json
from numpy import dtype, array, allclose
import os
import struct
from nose.tools import assert_equals, assert_true, assert_almost_equal

from thunder.rdds.fileio.seriesloader import SeriesLoader
from test_utils import PySparkTestCaseWithOutputDir


class SeriesBinaryTestData(object):
    """
    Data object for SeriesLoader binary test.
    """
    __slots__ = ('keys', 'vals', 'keyDType', 'valDType')

    def __init__(self, keys, vals, keydtype, valdtype):
        """
        Constructor, intended to be called from fromArrays class factory method.

        Expects m x n and m x p data for keys and vals.

        Parameters
        ----------
        keys: two dimensional array or sequence
        vals: two dimensional array or sequence
        keydtype: object castable to numpy dtype
            data type of keys
        valdtype: object castable to numpy dtype
            data type of values

        Returns
        -------
        self: new instance of SeriesBinaryTestData
        """
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

    def writeToFile(self, f):
        """
        Writes own key, value data to passed file handle in binary format
        Parameters
        ----------
        f: file handle, open for writing
            f will remain open after this call
        """
        for keys, vals in self.data:
            f.write(struct.pack(self.keyStructFormat, *keys))
            f.write(struct.pack(self.valStructFormat, *vals))

    @staticmethod
    def _validateLengths(dat):
        l = len(dat[0])
        for d in dat:
            assert len(d) == l, "Data of unequal lengths, %d and %d" % (l, len(d))

    @staticmethod
    def _normalizeDType(dtypeinst, data):
        if dtypeinst is None:
            return data.dtype
        return dtype(dtypeinst)

    @classmethod
    def fromArrays(cls, keys, vals, keydtype=None, valdtype=None):
        """
        Factory method for SeriesBinaryTestData. Validates input before calling class __init__ method.

        Expects m x n and m x p data for keys and vals.

        Parameters
        ----------
        keys: two dimensional array or sequence
        vals: two dimensional array or sequence
        keydtype: object castable to numpy dtype
            data type of keys
        valdtype: object castable to numpy dtype
            data type of values

        Returns
        -------
        self: new instance of SeriesBinaryTestData
        """
        keydtype = cls._normalizeDType(keydtype, keys)
        valdtype = cls._normalizeDType(valdtype, vals)
        assert len(keys) == len(vals), "Unequal numbers of keys and values, %d and %d" % (len(keys), len(vals))
        cls._validateLengths(keys)
        cls._validateLengths(vals)
        return cls(keys, vals, keydtype, valdtype)


class TestSeriesBinaryLoader(PySparkTestCaseWithOutputDir):

    def _run_tst_fromBinary(self, useConfJson=False):
        # run this as a single big test so as to avoid repeated setUp and tearDown of the spark context
        DATA = []
        # data will be a sequence of test data
        # all keys and all values in a test data item must be of the same length
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int16'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3], [5, 6, 7]], [[11], [12]], 'int16', 'int16'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int32'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int32', 'int16'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11.0, 12.0, 13.0]], 'int16', 'float32'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1.0, 2.0, 3.0]], [[11.0, 12.0, 13.0]], 'float32', 'float32'))
        DATA.append(SeriesBinaryTestData.fromArrays([[1.5, 2.5, 3.5]], [[11.0, 12.0, 13.0]], 'float32', 'float32'))

        for itemidx, item in enumerate(DATA):
            outsubdir = os.path.join(self.outputdir, 'input%d' % itemidx)
            os.mkdir(outsubdir)

            fname = os.path.join(outsubdir, 'inputfile%d.bin' % itemidx)
            with open(fname, 'wb') as f:
                item.writeToFile(f)

            loader = SeriesLoader(self.sc)
            if not useConfJson:
                series = loader.fromBinary(outsubdir, nkeys=item.nkeys, nvalues=item.nvals, keytype=str(item.keyDType),
                                           valuetype=str(item.valDType))
            else:
                # write configuration file
                conf = {'input': outsubdir,
                        'nkeys': item.nkeys, 'nvalues': item.nvals,
                        'format': str(item.valDType), 'keyformat': str(item.keyDType)}
                with open(os.path.join(outsubdir, "conf.json"), 'wb') as f:
                    json.dump(conf, f, indent=2)
                series = loader.fromBinary(outsubdir)

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
                assert_equals(item.valDType, actual[1].dtype,
                              "Value type mismatch in item %d; expected %s, got %s" %
                              (itemidx, str(item.valDType), str(actual[1].dtype)))

    def test_fromBinary(self):
        self._run_tst_fromBinary()

    def test_fromBinaryWithConfFile(self):
        self._run_tst_fromBinary(True)
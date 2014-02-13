import shutil
import tempfile
from numpy import array, allclose
from thunder.util.load import subtoind, indtosub, getdims
from test_utils import PySparkTestCase


class LoadTestCase(PySparkTestCase):
    def setUp(self):
        super(LoadTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(LoadTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestSubToInd(LoadTestCase):
    """Test conversion between linear and subscript indexing"""

    def test_sub_to_ind(self):
        subs = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)]
        data_local = map(lambda x: (x, array([1.0])), subs)

        data = self.sc.parallelize(data_local)
        dims = [2, 3, 2]
        inds = subtoind(data, dims).map(lambda (k, _): k).collect()
        assert(allclose(inds, array(range(1, 13))))

    def test_ind_to_sub(self):
        data_local = map(lambda x: (x, array([1.0])), range(1, 13))

        data = self.sc.parallelize(data_local)
        dims = [2, 3, 2]
        #subs = data.map(lambda (k, _): indtosub(k, dims)).collect()
        subs = indtosub(data, dims).map(lambda (k, _): k).collect()
        assert(allclose(subs, array([(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                                     (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)])))


class TestGetDims(LoadTestCase):
    """Test getting dimensions"""

    def test_get_dims(self):
        subs = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)]
        data_local = map(lambda x: (x, array([1.0])), subs)
        data = self.sc.parallelize(data_local)
        dims = getdims(data)
        assert(dims, array([2, 3, 2]))
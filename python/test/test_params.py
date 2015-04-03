from thunder.utils.params import Params
from nose.tools import assert_true
from numpy import array, array_equal

class TestParams:
    
    def test_paramsMethods(self):

        param1 = {'name': 'p1',
                  'value': array([1, 2, 3])}
        param2 = {'name': 'p2',
                  'value': array([4, 5, 6])}
        params = Params([param1, param2])

        target1 = array([1, 2, 3])
        target = array([[1, 2, 3], [4, 5, 6]])

        assert_true(params.names() == ['p1', 'p2'])
        assert_true(array_equal(params.values(), target))

        assert_true(array_equal(params.values('p1'), target1))
        assert_true(array_equal(params.values(['p1', 'p2']), target))
        assert_true(array_equal(params.values(('p1', 'p2')), target))

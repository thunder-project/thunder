import pytest
from numpy import array, allclose, corrcoef
from thunder.data.series import fromList

pytestmark = pytest.mark.usefixtures("context")


def test_fourier():
    data = fromList([array([1.0, 2.0, -4.0, 5.0, 8.0, 3.0, 4.1, 0.9, 2.3])])
    vals = data.totimeseries().fourier(freq=2)
    assert allclose(vals.select('coherence').toarray(), 0.578664)
    assert allclose(vals.select('phase').toarray(), 4.102501)


def test_convolve():
    data = fromList([array([1, 2, 3, 4, 5])])
    sig = array([1, 2, 3])
    betas = data.totimeseries().convolve(sig, mode='same')
    assert allclose(betas.toarray(), array([4, 10, 16, 22, 22]))


def test_crosscorr():
    local = array([1.0, 2.0, -4.0, 5.0, 8.0, 3.0, 4.1, 0.9, 2.3])
    data = fromList([local]).totimeseries()
    sig = array([1.5, 2.1, -4.2, 5.6, 8.1, 3.9, 4.2, 0.3, 2.1])
    betas = data.crosscorr(signal=sig, lag=0)
    assert allclose(betas.toarray(), corrcoef(local, sig)[0, 1])
    betas = data.crosscorr(signal=sig, lag=2)
    truth = array([-0.18511, 0.03817, 0.99221, 0.06567, -0.25750])
    assert allclose(betas.toarray(), truth, atol=1E-5)


def test_detrend():
    data = fromList([array([1, 2, 3, 4, 5])]).totimeseries()
    out = data.detrend('linear')
    assert allclose(out.toarray(), array([0, 0, 0, 0, 0]))


def test_normalize_percentile():
    data = fromList([array([1, 2, 3, 4, 5])]).totimeseries()
    out = data.normalize('percentile', perc=20)
    vals = out.toarray()
    assert str(vals.dtype) == 'float64'
    assert allclose(vals, array([-0.42105,  0.10526,  0.63157,  1.15789,  1.68421]), atol=1e-3)


def test_normalize_window():
    y = array([1, 2, 3, 4, 5])
    data = fromList([y]).totimeseries()
    vals = data.normalize('window', window=2).toarray()
    b = array([1, 1, 2, 3, 4])
    result_true = (y - b) / (b + 0.1)
    assert allclose(vals, result_true, atol=1e-3)
    vals = data.normalize('window', window=5).toarray()
    b = array([1, 1, 2, 3, 4])
    result_true = (y - b) / (b + 0.1)
    assert allclose(vals, result_true, atol=1e-3)


def test_normalize_window_exact():
    y = array([1, 2, 3, 4, 5])
    data = fromList([y]).totimeseries()
    vals = data.normalize('window-exact', window=2).toarray()
    b = array([1.2,  1.4,  2.4,  3.4,  4.2])
    result_true = (y - b) / (b + 0.1)
    assert allclose(vals, result_true, atol=1e-3)
    vals = data.normalize('window-exact', window=6).toarray()
    b = array([1.6,  1.8,  1.8,  1.8,  2.6])
    result_true = (y - b) / (b + 0.1)
    assert allclose(vals, result_true, atol=1e-3)


def test_normalize_mean():
    data = fromList([array([1, 2, 3, 4, 5])]).totimeseries()
    vals = data.normalize('mean').toarray()
    assert allclose(vals, array([-0.64516,  -0.32258,  0.0,  0.32258,  0.64516]), atol=1e-3)


def test_mean_by_window():
    data = fromList([array([0, 1, 2, 3, 4, 5, 6])]).totimeseries()
    test1 = data.mean_by_window(indices=[3, 5], window=2).toarray()
    assert allclose(test1, [3, 4])
    test2 = data.mean_by_window(indices=[3, 5], window=3).toarray()
    assert allclose(test2, [3, 4, 5])
    test3 = data.mean_by_window(indices=[3, 5], window=4).toarray()
    assert allclose(test3, [2, 3, 4, 5])
    test4 = data.mean_by_window(indices=[3], window=4).toarray()
    assert allclose(test4, [1, 2, 3, 4])


def test_group_by_window():
    data = fromList([array([0, 1, 2, 3, 4, 5, 6])]).totimeseries()
    test1 = data.group_by_window(indices=[3, 5], window=2).toarray()
    assert allclose(test1, [[2, 3], [4, 5]])
    test2 = data.group_by_window(indices=[3, 5], window=3).toarray()
    assert allclose(test2, [[2, 3, 4], [4, 5, 6]])
    test3 = data.group_by_window(indices=[3, 5], window=4).toarray()
    assert allclose(test3, [[1, 2, 3, 4], [3, 4, 5, 6]])
    test4 = data.group_by_window(indices=[3, 4, 5], window=2).toarray()
    assert allclose(test4, [[2, 3], [3, 4], [4, 5]])
    test5 = data.group_by_window(indices=[3], window=2).toarray()
    assert allclose(test5, [[2, 3]])
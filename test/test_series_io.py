import pytest
import os
import glob
from numpy import arange, array, allclose, save, savetxt
from scipy.io import savemat

from thunder.data.series.readers import fromarray, fromnpy, frommat, fromtext, frombinary, fromexample

pytestmark = pytest.mark.usefixtures("context")


def test_from_array():
    a = arange(8, dtype='int16').reshape((4, 2))
    data = fromarray(a)
    assert data.nrecords == 4
    assert data.dtype == 'int16'
    assert allclose(data.index, [0, 1])
    assert allclose(data.toarray(), a)


def test_from_array_index():
    a = arange(8, dtype='int16').reshape((4, 2))
    data = fromarray(a, index=[2, 3])
    assert allclose(data.index, [2, 3])


def test_from_npy(tmpdir):
    a = arange(8, dtype='int16').reshape((4, 2))
    f = os.path.join(str(tmpdir), 'data.npy')
    save(f, a)
    data = fromnpy(f)
    assert data.nrecords == 4
    assert data.dtype == 'int16'
    assert allclose(data.index, [0, 1])
    assert allclose(data.toarray(), a)


def test_from_mat(tmpdir):
    a = arange(8, dtype='int16').reshape((4, 2))
    f = os.path.join(str(tmpdir), 'data.mat')
    savemat(f, {'var': a})
    data = frommat(f, 'var')
    assert data.nrecords == 4
    assert data.dtype == 'int16'
    assert allclose(data.index, [0, 1])
    assert allclose(data.toarray(), a)


def test_from_text(tmpdir):
    k = [[i] for i in range(10)]
    v = [[0, i] for i in range(10)]
    a = [kv[0] + kv[1] for kv in zip(k, v)]
    f = os.path.join(str(tmpdir), 'data.txt')
    savetxt(f, a, fmt='%.02g')
    data = fromtext(f, nkeys=1)
    assert data.nrecords == 10
    assert data.dtype == 'float64'
    assert allclose(data.keys().collect(), k)
    assert allclose(data.values().collect(), v)


def test_from_binary(tmpdir):
    a = arange(8, dtype='int16').reshape((4, 2))
    p = os.path.join(str(tmpdir), 'data.bin')
    with open(p, 'w') as f:
        f.write(a)
    data = frombinary(p, nkeys=0, nvalues=2, keyType='int16', valueType='int16')
    assert data.nrecords == 4
    assert allclose(data.keys().collect(), [(i,) for i in range(4)])
    assert allclose(data.values().collect(), a)


def test_from_binary_keys(tmpdir):
    k = [[i] for i in range(10)]
    v = [[0, i] for i in range(10)]
    a = array([kv[0] + kv[1] for kv in zip(k, v)], dtype='int16')
    p = os.path.join(str(tmpdir), 'data.bin')
    with open(p, 'w') as f:
        f.write(a)
    data = frombinary(p, nkeys=1, nvalues=2, keyType='int16', valueType='int16')
    assert data.nrecords == 10
    assert allclose(data.keys().collect(), k)
    assert allclose(data.values().collect(), v)


def test_to_binary(tmpdir):
    a = arange(8, dtype='int16').reshape((4, 2))
    p = str(tmpdir) + '/data'
    fromarray(a, npartitions=1).tobinary(p)
    files = [os.path.basename(f) for f in glob.glob(str(tmpdir) + '/data/*')]
    assert sorted(files) == ['SUCCESS', 'conf.json', 'key00_00000.bin']


def test_to_binary_roundtrip(tmpdir):
    a = arange(8, dtype='int16').reshape((4, 2))
    p = str(tmpdir) + '/data'
    data = fromarray(a, npartitions=1)
    data.tobinary(p)
    loaded = frombinary(p)
    assert allclose(data.toarray(), loaded.toarray())


def test_fromexample():
    for d in ['fish', 'mouse', 'iris']:
        fromexample(d).count()
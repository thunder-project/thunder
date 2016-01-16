import pytest
import os
import glob
import json
from numpy import arange, array, allclose, save, savetxt
from scipy.io import savemat

from thunder.series.readers import fromarray, fromnpy, frommat, fromtext, frombinary, fromexample

pytestmark = pytest.mark.usefixtures("eng")


def test_from_array(eng):
    a = arange(8, dtype='int16').reshape((4, 2))
    data = fromarray(a, engine=eng)
    assert data.shape == (4, 2)
    assert data.dtype == 'int16'
    assert allclose(data.index, [0, 1])
    assert allclose(data.toarray(), a)


def test_from_array_vector(eng):
    a = arange(8, dtype='int16').reshape((4, 2))
    data = fromarray(a, engine=eng)
    assert data.shape == (4, 2)
    assert data.dtype == 'int16'
    assert allclose(data.index, [0, 1])
    assert allclose(data.toarray(), a)


def test_from_array_index(eng):
    a = arange(8, dtype='int16').reshape((4, 2))
    data = fromarray(a, index=[2, 3], engine=eng)
    assert allclose(data.index, [2, 3])


def test_from_npy(tmpdir, eng):
    a = arange(8, dtype='int16').reshape((4, 2))
    f = os.path.join(str(tmpdir), 'data.npy')
    save(f, a)
    data = fromnpy(f, engine=eng)
    assert data.shape == (4, 2)
    assert data.dtype == 'int16'
    assert allclose(data.index, [0, 1])
    assert allclose(data.toarray(), a)


def test_from_mat(tmpdir, eng):
    a = arange(8, dtype='int16').reshape((4, 2))
    f = os.path.join(str(tmpdir), 'data.mat')
    savemat(f, {'var': a})
    data = frommat(f, 'var', engine=eng)
    assert data.shape == (4, 2)
    assert data.dtype == 'int16'
    assert allclose(data.index, [0, 1])
    assert allclose(data.toarray(), a)


def test_from_text(tmpdir, eng):
    v = [[0, i] for i in range(10)]
    f = os.path.join(str(tmpdir), 'data.txt')
    savetxt(f, v, fmt='%.02g')
    data = fromtext(f, engine=eng)
    assert allclose(data.shape, (10, 2))
    assert data.dtype == 'float64'
    assert allclose(data.toarray(), v)


def test_from_text_skip(tmpdir):
    k = [[i] for i in range(10)]
    v = [[0, i] for i in range(10)]
    a = [kv[0] + kv[1] for kv in zip(k, v)]
    f = os.path.join(str(tmpdir), 'data.txt')
    savetxt(f, a, fmt='%.02g')
    data = fromtext(f, skip=1)
    assert allclose(data.shape, (10, 2))
    assert data.dtype == 'float64'
    assert allclose(data.toarray(), v)


def test_from_binary(tmpdir, eng):
    a = arange(8, dtype='int16').reshape((4, 2))
    p = os.path.join(str(tmpdir), 'data.bin')
    with open(p, 'w') as f:
        f.write(a)
    data = frombinary(p, shape=[4, 2], dtype='int16', engine=eng)
    assert allclose(data.shape, (4, 2))
    assert allclose(data.index, [0, 1])
    assert allclose(data.toarray(), a)


def test_from_binary_skip(tmpdir, eng):
    k = [[i] for i in range(10)]
    v = [[0, i] for i in range(10)]
    a = array([kv[0] + kv[1] for kv in zip(k, v)], dtype='int16')
    p = os.path.join(str(tmpdir), 'data.bin')
    with open(p, 'w') as f:
        f.write(a)
    data = frombinary(p, shape=[10, 2], dtype='int16', skip=1, engine=eng)
    assert allclose(data.shape, (10, 2))
    assert allclose(data.index, [0, 1])
    assert allclose(data.toarray(), v)


def test_to_binary(tmpdir, eng):
    a = arange(8, dtype='int16').reshape((4, 2))
    p = str(tmpdir) + '/data'
    fromarray(a, npartitions=1, engine=eng).tobinary(p)
    files = [os.path.basename(f) for f in glob.glob(str(tmpdir) + '/data/*')]
    assert sorted(files) == ['SUCCESS', 'conf.json', 'key00_00000.bin']
    with open(str(tmpdir) + '/data/conf.json', 'r') as f:
        conf = json.load(f)
        assert conf['shape'] == [4, 2]
        assert conf['dtype'] == 'int16'


def test_to_binary_roundtrip(tmpdir, eng):
    a = arange(8, dtype='int16').reshape((4, 2))
    p = str(tmpdir) + '/data'
    data = fromarray(a, npartitions=1, engine=eng)
    data.tobinary(p)
    loaded = frombinary(p)
    assert allclose(data.toarray(), loaded.toarray())


def test_to_binary_roundtrip_3d(tmpdir, eng):
    a = arange(16, dtype='int16').reshape((4, 2, 2))
    p = str(tmpdir) + '/data'
    data = fromarray(a, npartitions=1, engine=eng)
    data.tobinary(p)
    loaded = frombinary(p, engine=eng)
    assert allclose(data.toarray(), loaded.toarray())


def test_from_example(eng):
    data = fromexample('fish', engine=eng)
    assert allclose(data.shape, (76, 87, 2, 240))
    data = fromexample('mouse', engine=eng)
    assert allclose(data.shape, (64, 64, 500))
    data = fromexample('iris', engine=eng)
    assert allclose(data.shape, (150, 4))
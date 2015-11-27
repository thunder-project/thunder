import pytest
import os
import glob
import json
from numpy import arange, allclose

from thunder.data.images.readers import fromList, fromArray, fromPng, fromTif, fromBinary

pytestmark = pytest.mark.usefixtures("context")

resources = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')


def test_from_list():
    a = arange(8).reshape((2, 4))
    data = fromList([a])
    vals = data.collect()
    assert len(vals) == 1
    assert allclose(data.dims, a.shape)
    assert allclose(data.toarray(), a)


def test_from_array():
    a = arange(8).reshape((1, 2, 4))
    data = fromArray(a)
    vals = data.collect()
    assert len(vals) == 1
    assert allclose(data.dims, a[0].shape)
    assert allclose(data.toarray(), a)


def test_from_png():
    path = os.path.join(resources, 'singlelayer_png', 'dot1_grey.png')
    data = fromPng(path)
    assert data.count() == 1
    assert allclose(data.dims, (70, 75))
    assert allclose(data.values().first().shape, (70, 75))
    assert allclose(data.values().first().max(), 239)
    assert allclose(data.values().first().min(), 1)


def test_from_tif():
    path = os.path.join(resources, "singlelayer_tif", "dot1_grey_lzw.tif")
    data = fromTif(path)
    assert data.count() == 1
    assert allclose(data.dims, (70, 75))
    assert allclose(data.values().first().shape, (70, 75))
    assert allclose(data.values().first().max(), 239)
    assert allclose(data.values().first().min(), 1)


def test_from_tif_many():
    path = os.path.join(resources, "singlelayer_tif", "dot*_grey_lzw.tif")
    data = fromTif(path)
    assert data.count() == 3
    assert allclose(data.dims, (70, 75))
    assert allclose(data.values().first().shape, (70, 75))
    assert [x.sum() for x in data.values().collect()] == [1233881, 1212169, 1191300]


def test_from_tif_multi_lzw():
    path = os.path.join(resources, 'multilayer_tif', 'dotdotdot_lzw.tif')
    data = fromTif(path)
    val = data.values().first()
    assert data.count() == 1
    assert allclose(data.dims, (70, 75, 3))
    assert allclose(data.values().first().shape, (70, 75, 3))
    assert [val[:, :, i].sum() for i in range(3)] == [1140006, 1119161, 1098917]


def test_from_tif_multi_float():
    path = os.path.join(resources, 'multilayer_tif', 'dotdotdot_float32.tif')
    data = fromTif(path)
    val = data.values().first()
    assert data.count() == 1
    assert allclose(data.dims, (70, 75, 3))
    assert allclose(data.values().first().shape, (70, 75, 3))
    assert [val[:, :, i].sum() for i in range(3)] == [1140006, 1119161, 1098917]


def test_from_tif_multi_planes():
    path = os.path.join(resources, 'multilayer_tif', 'dotdotdot_lzw.tif')
    data = fromTif(path, nplanes=3)
    assert data.count() == 1
    data = fromTif(path, nplanes=1)
    assert data.count() == 3
    assert allclose(data.keys().collect(), [0, 1, 2])
    assert allclose(data.values().first().shape, (70, 75))
    assert [x.sum() for x in data.values().collect()] == [1140006, 1119161, 1098917]


def test_from_tif_multi_planes_many():
    path = os.path.join(resources, 'multilayer_tif', 'dotdotdot_lzw*.tif')
    data = fromTif(path, nplanes=3)
    assert data.nrecords == 2
    assert allclose(data.keys().collect(), [0, 1])
    assert allclose(data.dims, (70, 75, 3))
    assert allclose(data.values().first().shape, (70, 75, 3))
    data = fromTif(path, nplanes=1)
    assert data.nrecords == 6
    assert allclose(data.keys().collect(), [0, 1, 2, 3, 4, 5])
    assert allclose(data.dims, (70, 75))
    assert allclose(data.values().first().shape, (70, 75))
    assert [x.sum() for x in data.values().collect()] == [1140006, 1119161, 1098917, 1140006, 1119161, 1098917]


def test_from_tif_signed():
    path = os.path.join(resources, 'multilayer_tif', 'test_signed.tif')
    data = fromTif(path, nplanes=1)
    assert data.count() == 2
    assert data.dtype == 'int16'
    assert allclose(data.values().first().shape, (120, 120))
    assert [x.sum() for x in data.values().collect()] == [1973201, 2254767]


def test_from_binary(tmpdir):
    a = arange(8, dtype='int16').reshape((2, 4))
    a.tofile(os.path.join(str(tmpdir), 'test.bin'))
    data = fromBinary(str(tmpdir), dims=(2, 4), dtype='int16')
    assert data.nrecords == 1
    assert data.dtype == 'int16'
    assert allclose(data.dims, (2, 4))
    assert allclose(data.values().first(), a)


def test_from_binary_many(tmpdir):
    a = [arange(8, dtype='int16').reshape((2, 4)), arange(8, 16, dtype='int16').reshape((2, 4))]
    a[0].tofile(os.path.join(str(tmpdir), 'test0.bin'))
    a[1].tofile(os.path.join(str(tmpdir), 'test1.bin'))
    data = fromBinary(str(tmpdir), dims=(2, 4), dtype='int16')
    assert data.nrecords == 2
    assert data.dtype == 'int16'
    assert allclose(data.dims, (2, 4))
    assert allclose(data.toarray(), a)


def test_from_binary_conf(tmpdir):
    a = [arange(8, dtype='int32').reshape((2, 4)), arange(8, 16, dtype='int32').reshape((2, 4))]
    a[0].tofile(os.path.join(str(tmpdir), 'test0.bin'))
    a[1].tofile(os.path.join(str(tmpdir), 'test1.bin'))
    with open(os.path.join(str(tmpdir), 'conf.json'), 'w') as f:
        json.dump({'dims': [2, 4], 'dtype': 'int32'}, f)
    data = fromBinary(str(tmpdir))
    assert data.nrecords == 2
    assert data.dtype == 'int32'
    assert allclose(data.dims, (2, 4))
    assert allclose(data.toarray(), a)


def test_from_binary_multi(tmpdir):
    a = arange(24, dtype='int16').reshape((2, 3, 4))
    a.tofile(os.path.join(str(tmpdir), 'test.bin'))
    data = fromBinary(str(tmpdir), dims=(2, 3, 4), dtype='int16')
    assert data.nrecords == 1
    assert data.dtype == 'int16'
    assert allclose(data.dims, (2, 3, 4))
    assert allclose(data.values().first(), a)


def test_from_binary_multi_planes_many(tmpdir):
    a = [arange(16, dtype='int16').reshape((4, 2, 2)), arange(16, 32, dtype='int16').reshape((4, 2, 2))]
    a[0].tofile(os.path.join(str(tmpdir), 'test0.bin'))
    a[1].tofile(os.path.join(str(tmpdir), 'test1.bin'))
    data = fromBinary(str(tmpdir), dims=(4, 2, 2), dtype='int16', nplanes=2)
    assert data.nrecords == 2
    assert allclose(data.dims, (4, 2, 2))
    assert allclose(data.values().first().shape, (4, 2, 2))
    data = fromBinary(str(tmpdir), dims=(4, 2, 2), dtype='int16', nplanes=1)
    assert data.nrecords == 4
    assert allclose(data.dims, (4, 2))
    assert allclose(data.values().first().shape, (4, 2))


def test_to_binary(tmpdir):
    a = [arange(8, dtype='int16').reshape((4, 2)), arange(8, 16, dtype='int16').reshape((4, 2))]
    fromList(a).toBinary(os.path.join(str(tmpdir), 'binary'), prefix='image')
    files = [os.path.basename(f) for f in glob.glob(str(tmpdir) + '/binary/image*')]
    f = open(str(tmpdir) + '/binary/conf.json', 'r')
    conf = json.load(f)
    f.close()
    assert sorted(files) == ['image-00000.bin', 'image-00001.bin']
    assert conf['dims'] == [4, 2]
    assert conf['dtype'] == 'int16'


def test_to_binary_roundtrip(tmpdir):
    a = [arange(8).reshape((4, 2)), arange(8, 16).reshape((4, 2))]
    data = fromList(a)
    data.toBinary(os.path.join(str(tmpdir), 'images'))
    loaded = fromBinary(os.path.join(str(tmpdir), 'images'))
    assert allclose(data.toarray(), loaded.toarray())


def test_to_binary_roundtrip_3d(tmpdir):
    a = [arange(24).reshape((2, 3, 4)), arange(24, 48).reshape((2, 3, 4))]
    data = fromList(a)
    data.toBinary(os.path.join(str(tmpdir), 'images'))
    loaded = fromBinary(os.path.join(str(tmpdir), 'images'))
    assert allclose(data.toarray(), loaded.toarray())


def test_to_png(tmpdir):
    a = [arange(8, dtype='int16').reshape((4, 2)), arange(8, 16, dtype='int16').reshape((4, 2))]
    fromList(a).toPng(os.path.join(str(tmpdir), 'images'), prefix='image')
    files = [os.path.basename(f) for f in glob.glob(str(tmpdir) + '/images/image*')]
    assert sorted(files) == ['image-00000.png', 'image-00001.png']


def test_to_png_roundtrip(tmpdir):
    a = [arange(8, dtype='uint8').reshape((4, 2))]
    data = fromList(a)
    data.toPng(os.path.join(str(tmpdir), 'images'), prefix='image')
    loaded = fromPng(os.path.join(str(tmpdir), 'images'))
    assert allclose(data.toarray(), loaded.toarray())


def test_to_tif(tmpdir):
    a = [arange(8, dtype='int16').reshape((4, 2)), arange(8, 16, dtype='int16').reshape((4, 2))]
    fromList(a).toTif(os.path.join(str(tmpdir), 'images'), prefix='image')
    files = [os.path.basename(f) for f in glob.glob(str(tmpdir) + '/images/image*')]
    assert sorted(files) == ['image-00000.tif', 'image-00001.tif']


def test_to_tif_roundtrip(tmpdir):
    a = [arange(8, dtype='uint8').reshape((4, 2))]
    data = fromList(a)
    data.toTif(os.path.join(str(tmpdir), 'images'), prefix='image')
    loaded = fromTif(os.path.join(str(tmpdir), 'images'))
    assert allclose(data.toarray(), loaded.toarray())

from numpy import ones

from test_utils import PySparkTestCase

from thunder.rdds.fileio.imagesloader import ImagesLoader


class TestBlockKeys(PySparkTestCase):

    def setUp(self):
        super(TestBlockKeys, self).setUp()
        shape = (30, 30)
        arys = [ones(shape) for _ in range(0, 3)]
        data = ImagesLoader(self.sc).fromArrays(arys)
        self.blocks = data.toBlocks(size=(10, 10)).collect()
        self.keys = [k for k, v in self.blocks]

    def test_attributes(self):
        """
        (TestBlockKeys) attributes
        """
        assert(all(k.pixelsPerDim == (10, 10) for k in self.keys))
        assert(all(k.spatialShape == (10, 10) for k in self.keys))

    def test_neighbors(self):
        """
        (TestBlockKeys) neighbors
        """
        keys = self.keys
        assert(keys[0].neighbors() == [(0, 10), (10, 0), (10, 10)])
        assert(keys[1].neighbors() == [(0, 0), (0, 10), (10, 10), (20, 0), (20, 10)])
        assert(keys[2].neighbors() == [(10, 0), (10, 10), (20, 10)])
        assert(keys[3].neighbors() == [(0, 0), (0, 20), (10, 0), (10, 10), (10, 20)])
        assert(keys[4].neighbors() == [(0, 0), (0, 10), (0, 20), (10, 0), (10, 20), (20, 0), (20, 10), (20, 20)])
        assert(keys[5].neighbors() == [(10, 0), (10, 10), (10, 20), (20, 0), (20, 20)])
        assert(keys[6].neighbors() == [(0, 10), (10, 10), (10, 20)])
        assert(keys[7].neighbors() == [(0, 10), (0, 20), (10, 10), (20, 10), (20, 20)])
        assert(keys[8].neighbors() == [(10, 10), (10, 20), (20, 10)])
from numpy import array_equal, sum, zeros, where, asarray, squeeze

from thunder.extraction import BasicBlockMerger, OverlapBlockMerger
from thunder.extraction.source import Source
from thunder.rdds.fileio.imagesloader import ImagesLoader

from test_utils import PySparkTestCase


class TestBlockMerger(PySparkTestCase):

    def test_basic_merger(self):
        """
        (BlockMerger) basic merger
        """
        alg = BasicBlockMerger()

        # center circle, 8 neighbors
        sources, keys, mask = self.generateSources(padding=3)
        m = alg.merge(sources, keys)
        assert(m.count == 9)
        assert(array_equal(m.masks((30, 30)), mask))

        # left circle, 5 neighbors
        sources, keys, mask = self.generateSources(center=(15, 5), padding=3)
        m = alg.merge(sources, keys)
        assert(m.count == 6)
        assert(array_equal(m.masks((30, 30)), mask))

        # can run with or without padding
        sources, keys, mask = self.generateSources(padding=3)
        m1 = alg.merge(sources, keys)
        sources, keys, mask = self.generateSources(padding=0)
        m2 = alg.merge(sources, keys)
        assert(m1.count == 9)
        assert(m2.count == 3)
        assert(array_equal(m1.masks((30, 30)), mask))
        assert(array_equal(m2.masks((30, 30)), mask))

    def test_overlap_merger(self):
        """
        (BlockMerger) overlap merger
        """
        # center circle, 8 neighbors, padding 7
        sources, keys, mask = self.generateSources(padding=7)
        alg = OverlapBlockMerger(overlap=0.5)
        m = alg.merge(sources, keys)
        assert(m.count == 1)
        assert(array_equal(m[0].mask((30, 30)), mask))

        # left center
        sources, keys, mask = self.generateSources(center=(15, 5), padding=7)
        alg = OverlapBlockMerger(overlap=0.5)
        m = alg.merge(sources, keys)
        assert(m.count == 1)
        assert(array_equal(m[0].mask((30, 30)), mask))

        # right center
        sources, keys, mask = self.generateSources(center=(15, 24), padding=7)
        alg = OverlapBlockMerger(overlap=0.5)
        m = alg.merge(sources, keys)
        assert(m.count == 1)
        assert(array_equal(m[0].mask((30, 30)), mask))

    def test_overlap_merger_thresholds(self):
        """
        (BlockMerger) overlap merger, varying thresholds
        """
        # center circle, 8 neighbors, padding 3
        sources, keys, mask = self.generateSources(padding=3)

        # low threshold, all merged
        alg = OverlapBlockMerger(overlap=0.1)
        m = alg.merge(sources, keys)
        assert(m.count == 1)
        assert(array_equal(m[0].mask((30, 30)), mask))

        # high threshold, none merged
        alg = OverlapBlockMerger(overlap=0.5)
        m = alg.merge(sources, keys)
        assert(m.count == 9)
        assert(array_equal(m.masks((30, 30)), mask))

        # center circle, 8 neighbors, padding 5
        # higher threshold, still all merged
        sources, keys, mask = self.generateSources(padding=5)
        alg = OverlapBlockMerger(overlap=0.2)
        m = alg.merge(sources, keys)
        assert(m.count == 1)
        assert(array_equal(m[0].mask((30, 30)), mask))

    def generateSources(self, padding, center=(15, 15), radius=6):
        """
        Generate a set of sources and block keys
        by constructing a circular mask region,
        generating blocks (with or without padding),
        and returning the sources defined by the mask
        in each block, and the block keys
        """
        from skimage.draw import circle

        mask = zeros((30, 30))
        rr, cc = circle(center[0], center[1], radius)
        mask[rr, cc] = 1
        img = ImagesLoader(self.sc).fromArrays([mask])
        blks = img.toBlocks(size=(10, 10), padding=padding).collect()
        keys, vals = zip(*blks)
        sources = [[Source(asarray(where(squeeze(v))).T)] if sum(v) > 0 else [] for v in vals]

        return sources, keys, mask
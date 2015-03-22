from thunder.extraction.block.base import BlockMethod, BlockAlgorithm
from thunder.extraction.source import Source


class BlockNMF(BlockMethod):

    def __init__(self, **kwargs):
        algorithm = NMFBlockAlgorithm(**kwargs)
        super(self.__class__, self).__init__(algorithm, **kwargs)


class NMFBlockAlgorithm(BlockAlgorithm):
    """
    Find sources using non-negative matrix factorization on blocks.

    NMF on each block provides a candidate set of basis functions.
    These are then converted into regions using simple morphological operators:
    labeling connected components, and removing all that fail to meet
    min and max size thresholds.

    Parameters
    ----------
    maxIter : int, optional, default = 10
        Maximum number of iterations

    componentsPerBlock : int, optional, deafut = 3
        Number of components to find per block
    """
    def __init__(self, maxIter=10, componentsPerBlock=3, percentile=75, minArea=50, maxArea="block", **extra):
        self.maxIter = maxIter
        self.componentsPerBlock = componentsPerBlock
        self.percentile = percentile
        self.minArea = minArea
        self.maxArea = maxArea

    def extract(self, block):
        from numpy import clip, inf, percentile, asarray, where, size, prod
        from sklearn.decomposition import NMF
        from skimage.measure import label
        from skimage.morphology import remove_small_objects

        # get dimensions
        n = self.componentsPerBlock
        dims = block.shape[1:]

        # handle maximum size
        if self.maxArea == "block":
            maxArea = prod(dims) / 2
        else:
            maxArea = self.maxArea

        # reshape to be t x all spatial dimensions
        data = block.reshape(block.shape[0], -1)

        # build and apply NMF model to block
        model = NMF(n, max_iter=self.maxIter)
        model.fit(clip(data, 0, inf))

        # reconstruct sources as spatial objects in one array
        comps = model.components_.reshape((n,) + dims)

        # convert from basis functions into shape
        # by finding connected components and removing small objects
        combined = []
        for c in comps:
            tmp = c > percentile(c, self.percentile)
            shape = remove_small_objects(label(tmp), min_size=self.minArea)
            coords = asarray(where(shape)).T
            if (size(coords) > 0) and (size(coords) < maxArea):
                combined.append(Source(coords))

        return combined



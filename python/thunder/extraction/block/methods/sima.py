from __future__ import absolute_import

from numpy import array, asarray, where

from thunder.extraction.block.base import BlockMethod, BlockAlgorithm
from thunder.extraction.source import Source


class BlockSIMA(BlockMethod):

    def __init__(self, simaStrategy=None, **kwargs):
        algorithm = SIMABlockAlgorithm(simaStrategy=simaStrategy, **kwargs)
        super(self.__class__, self).__init__(algorithm, **kwargs)


class SIMABlockAlgorithm(BlockAlgorithm):
    """
    Extract sources using a SIMA SegmentationStrategy.

    Parameters
    ----------
    simaStrategy : sima.segment.SegmentationStrategy
        The strategy from SIMA to be used.
    """
    def __init__(self, simaStrategy=None, **extra):

        import sima.segment

        if not isinstance(simaStrategy, sima.segment.SegmentationStrategy):
            raise TypeError("Must provide a SIMA segmentation strategy, got %s" % type(simaStrategy))

        self.strategy = simaStrategy

    def extract(self, block):

        import sima

        # reshape the block to (t, z, y, x, c)
        dims = block.shape
        if len(dims) == 3:  # (t, x, y)
            reshapedBlock = block.reshape(dims[0], 1, dims[2], dims[1], 1)
        else:  # (t, x, y, z)
            reshapedBlock = block.reshape(dims[0], dims[3], dims[2], dims[1], 1)

        # create SIMA dataset from block
        dataset = sima.ImagingDataset([sima.Sequence.create('ndarray', reshapedBlock)], None)

        # apply the sima strategy to the dataset
        rois = self.strategy.segment(dataset)

        # convert the coordinates between the SIMA and thunder conventions
        coords = [asarray(where(array(roi))).T for roi in rois]
        if len(dims) == 3:
            coords = [c[:, 1:] for c in coords]

        # format the sources
        sources = [Source(c) for c in coords]

        return sources

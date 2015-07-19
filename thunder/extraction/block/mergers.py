from thunder.extraction.block.base import BlockMerger
from thunder.extraction.source import SourceModel


class BasicBlockMerger(BlockMerger):
    """
    Simple merger that combines sources directly
    without any merging or modification
    """
    def __init__(self, **extra):
        pass

    def merge(self, blocks, keys):
        """
        Parameters
        ----------
        blocks : list of lists of sources
            List of the sources found for each block; every block
            should be represented, with an empty list for blocks
            without any identified sources

        keys : List of BlockGroupingKeys or PaddedBlockGroupingKeys
            The keys for each of the blocks assocaited with the sources

        Returns
        -------
        SourceModel containing the combined list of sources
        """
        import itertools
        from thunder.rdds.imgblocks.blocks import PaddedBlockGroupingKey

        # recenter coordinates using the spatial key from each block
        # also subtract off initial padding if blocks were padded
        for ib, blk in enumerate(blocks):

            for source in blk:
                source.coordinates += keys[ib].spatialKey

            if isinstance(keys[0], PaddedBlockGroupingKey):
                for source in blk:
                    source.coordinates -= keys[ib].padding[0]

        # flatten list of sources
        chain = itertools.chain.from_iterable(blocks)
        sources = list(chain)

        return SourceModel(sources)


class OverlapBlockMerger(BlockMerger):
    """
    Merger that combines sources across blocks by
    merging sources that overlap with sources in neighboring blocks

    Parameters
    ----------
    overlap : scalar, optional, default = 0.5
        Degree of overlap requires for sources to be merged
    """
    def __init__(self, overlap=0.5, **extra):
        self.overlap = overlap

    def merge(self, blocks, keys):
        """
        Parameters
        ----------
        blocks : list of lists of sources
            List of the sources found for each block; every block
            should be represented, with an empty list for blocks
            without any identified sources

        keys : List of PaddedBlockGroupingKeys
            The keys for each of the blocks assocaited with the sources

        Returns
        -------
        SourceModel containing the combined list of merged sources
        """
        import itertools
        import copy
        from thunder.rdds.imgblocks.blocks import PaddedBlockGroupingKey

        # check that keys are from padded blocks
        if not all(isinstance(k, PaddedBlockGroupingKey) for k in keys):
            raise ValueError("All keys must correspond to padded blocks for this merger")

        blocks = copy.deepcopy(blocks)

        # re-center coordinates
        for ib, blk in enumerate(blocks):
            for source in blk:
                source.coordinates += keys[ib].spatialKey
                source.coordinates -= keys[ib].padding[0]

        # create lookup table from spatial keys -> block indices
        d = {}
        for i, k in enumerate(keys):
            d[k.spatialKey] = i

        # for all the sources in each block,
        # merge and delete its neighbors if
        # - they exceed an overlap threshold, and
        # - they are smaller than the current source
        for ib, blk in enumerate(blocks):
            neighbors = keys[ib].neighbors()
            for source in blk:
                for key in neighbors:
                    ind = d[key]
                    for other in blocks[ind][:]:
                        if source.overlap(other) > self.overlap and other.area < source.area:
                            source.merge(other)
                            blocks[ind].remove(other)

        chain = itertools.chain.from_iterable(blocks)
        sources = list(chain)

        return SourceModel(sources)

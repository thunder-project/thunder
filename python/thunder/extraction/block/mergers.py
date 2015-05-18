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
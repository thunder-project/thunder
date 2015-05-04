from thunder.extraction.block.base import BlockMerger
from thunder.extraction.source import SourceModel


class BasicBlockMerger(BlockMerger):

    def __init__(self, **extra):
        pass

    def merge(self, blocks, keys, data=None):
        import itertools
        from thunder.rdds.imgblocks.blocks import PaddedBlockGroupingKey

        # recenter coordinates using the spatial key from each block
        # also subtract off initial padding if blocks were padded
        for i, blk in enumerate(blocks):

            for source in blk:
                source.coordinates += keys[i].spatialKey

            if isinstance(keys[0], PaddedBlockGroupingKey):
                for source in blk:
                    source.coordinates -= keys[i].padding[0]

        # flatten list of sources
        chain = itertools.chain.from_iterable(blocks)
        sources = list(chain)

        return SourceModel(sources)


class PaddedBlockMerger(BlockMerger):

    def __init__(self, overlap=0.5, **extra):
        self.overlap = overlap

    def merge(self, blocks, keys, data=None):

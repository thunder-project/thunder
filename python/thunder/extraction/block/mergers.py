from thunder.extraction.block.base import BlockMerger
from thunder.extraction.source import SourceModel


class BasicBlockMerger(BlockMerger):

    def __init__(self, **extra):
        pass

    def merge(self, sources, keys, data=None):
        import itertools

        # recenter coordinates using the spatial key from each block
        for (i, sourceBlock) in enumerate(sources):
            for source in sourceBlock:
                source.coordinates += keys[i].spatialKey

        # flatten list of sources
        chain = itertools.chain.from_iterable(sources)
        sources = list(chain)
        return SourceModel(sources)
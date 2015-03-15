from thunder.extraction.block.base import BlockMerger


class BasicBlockMerger(BlockMerger):

    def __init__(self, overlap=0.5, **extra):
        self.overlap = overlap

    def merge(self, sources, data=None):
        return sources
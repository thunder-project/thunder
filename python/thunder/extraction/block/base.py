from thunder.extraction.extraction import SourceExtractionMethod
from thunder.rdds.images import Images
from thunder.rdds.imgblocks.blocks import Blocks, PaddedBlocks


class BlockMethod(SourceExtractionMethod):
    """
    Extract sources from spatiotemporal data by independently processing blocks.

    A block method extracts sources by applying a function to each block to recover
    sources, and then merging sources across blocks. It requires three components:
    an algorithm (which extracts sources from a single block),
    a merger (which combines sources across blocks),
    and a cleaner (which filters the output and removes or fixes undesired sources).

    Parameters
    ----------
    algorithm : BlockAlgorithm
        Which algorithm to use

    merger : BlockMerger, optional, default = BasicBlockMerger
        Which merger to use

    cleaner : Cleaner, optional, deafault = BasicCleaner
        Which cleaner to use

    kwargs : dict
        Any extra arguments will be passed to the algorithm, merger, and cleaner,
        useful for providing options to these components
    """
    def __init__(self, algorithm=None, merger=None, cleaner=None, **kwargs):

        from thunder.extraction.block.mergers import BasicBlockMerger
        from thunder.extraction.cleaners import BasicCleaner

        self.merger = merger if merger is not None else BasicBlockMerger(**kwargs)
        self.cleaner = cleaner if cleaner is not None else BasicCleaner(**kwargs)
        self.algorithm = algorithm

    def fit(self, blocks, size=(50, 50, 1), units="pixels", padding=0):
        """
        Fit the source extraction model to data

        Parameters
        ----------
        blocks : Blocks, PaddedBlocks, or Images
            Data in blocks, Images will automatically be converted to blocks

        size : tuple, optional, default = (50,50,1)
            Block size if converting from images

        units : string, optional, default = "pixels"
            Units for block size specification if converting from images

        padding : int or tuple, optional, deafult = 0
            Amount of padding if converting from images

        See also
        --------
        Images.toBlocks
        """
        if isinstance(blocks, Images):
            blocks = blocks.toBlocks(size, units, padding)

        elif not (isinstance(blocks, Blocks) or isinstance(blocks, PaddedBlocks)):
            raise Exception("Input must be Images, Blocks, or PaddedBlocks")

        if not isinstance(self.algorithm, BlockAlgorithm):
            raise Exception("A BlockAlgorithm must be specified")

        algorithm = self.algorithm

        parts = blocks.rdd.mapValues(algorithm.extract).collect()
        model = self.merger.merge(parts)

        if len(model.sources) < 1:
            raise Exception("No sources found, try changing parameters?")

        model = self.cleaner.clean(model)

        return model


class BlockAlgorithm(object):
    """
    Exposes an algorithm for extracting sources from a single block.
    """

    def extract(self, block):
        raise NotImplementedError


class BlockMerger(object):
    """
    Exposes a method for merging sources across blocks
    """

    def merge(self, sources, data=None):
        raise NotImplementedError

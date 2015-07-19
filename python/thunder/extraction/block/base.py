from thunder.extraction.extraction import SourceExtractionMethod
from thunder.rdds.images import Images
from thunder.rdds.imgblocks.blocks import Blocks, PaddedBlocks


class BlockMethod(SourceExtractionMethod):
    """
    Extract sources from spatiotemporal data by independently processing blocks.

    A block method extracts sources by applying a function to each block to recover
    sources, and then merging sources across blocks. It requires two components:
    an algorithm (which extracts sources from a single block) and
    a merger (which combines sources across blocks)

    Parameters
    ----------
    algorithm : BlockAlgorithm
        Which algorithm to use

    merger : BlockMerger, optional, default = BasicBlockMerger
        Which merger to use

    kwargs : dict
        Any extra arguments to be passed to the algorithm or merger,
        useful for providing options to these components
    """
    def __init__(self, algorithm=None, merger=None, **kwargs):

        from thunder.extraction.block.mergers import BasicBlockMerger

        self.merger = merger if merger is not None else BasicBlockMerger(**kwargs)
        self.algorithm = algorithm

    def fit(self, blocks, size=None, units="pixels", padding=0):
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
            if size is None:
                raise Exception("Must specify a size if images will be converted to blocks")
            blocks = blocks.toBlocks(size, units, padding)

        elif not (isinstance(blocks, Blocks) or isinstance(blocks, PaddedBlocks)):
            raise Exception("Input must be Images, Blocks, or PaddedBlocks")

        if not isinstance(self.algorithm, BlockAlgorithm):
            raise Exception("A BlockAlgorithm must be specified")

        algorithm = self.algorithm

        result = blocks.rdd.mapValues(algorithm.extract).collect()

        keys = map(lambda x: x[0], result)
        sources = map(lambda x: x[1], result)

        if sum(map(lambda b: len(b), sources)) < 1:
            raise Exception("No sources found, try changing parameters?")

        model = self.merger.merge(sources, keys)

        if len(model.sources) < 1:
            raise Exception("No sources found, try changing parameters?")

        return model


class BlockAlgorithm(object):
    """
    An algorithm for extracting sources from a single block.
    """
    def extract(self, block):
        raise NotImplementedError


class BlockMerger(object):
    """
    A method for merging sources across blocks
    """
    def merge(self, sources, keys):
        raise NotImplementedError

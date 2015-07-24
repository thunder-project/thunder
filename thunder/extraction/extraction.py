from thunder.utils.common import checkParams
from thunder.extraction.source import SourceModel


class SourceExtraction(object):
    """
    Factory for constructing source extraction methods.

    Returns a source extraction method given a string identifier.
    Options include: 'nmf', 'localmax', 'sima'
    """
    def __new__(cls, method, **kwargs):

        from thunder.extraction.block.methods.nmf import BlockNMF
        from thunder.extraction.block.methods.sima import BlockSIMA
        from thunder.extraction.feature.methods.localmax import LocalMax

        EXTRACTION_METHODS = {
            'nmf': BlockNMF,
            'localmax': LocalMax,
            'sima': BlockSIMA
        }

        checkParams(method, EXTRACTION_METHODS.keys())
        return EXTRACTION_METHODS[method](**kwargs)

    @staticmethod
    def load(file):
        return SourceModel.load(file)

    @staticmethod
    def deserialize(file):
        return SourceModel.deserialize(file)


class SourceExtractionMethod(object):

    def fit(self, data):
        raise NotImplementedError

    def run(self, data):

        model = self.fit(data)
        series = model.transform(data)

        return model, series

from thunder.utils.common import checkParams
from thunder.extraction.source import SourceModel


class SourceExtraction(object):

    def __new__(cls, method, **kwargs):

        from thunder.extraction.block.methods.nmf import BlockNMF
        from thunder.extraction.feature.methods.localmax import LocalMax

        EXTRACTION_METHODS = {
            'nmf': BlockNMF,
            'localmax': LocalMax
        }

        checkParams(method, EXTRACTION_METHODS.keys())
        return EXTRACTION_METHODS[method](**kwargs)

    @staticmethod
    def load(file):
        SourceModel.load(file)


class SourceExtractionMethod(object):

    def fit(self, data):
        raise NotImplementedError

    def run(self, data):

        model = self.fit(data)
        series = model.transform(data)

        return model, series

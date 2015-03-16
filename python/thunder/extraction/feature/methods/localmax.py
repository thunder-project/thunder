from thunder.extraction.feature.base import FeatureMethod, FeatureAlgorithm
from thunder.extraction.feature.preprocessors import MeanPreprocessor

class LocalMax(FeatureMethod):

    def __init__(self, **kwargs):
        algorithm = LocalMaxAlgorithm(**kwargs)
        preprocess = MeanPreprocessor()
        super(self.__class__, self).__init__(algorithm, preprocess, **kwargs)


class LocalMaxAlgorithm(FeatureAlgorithm):

    def __init__(self, threshold=10, **extra):
        self.threshold = threshold

    def extract(self, im):
        pass

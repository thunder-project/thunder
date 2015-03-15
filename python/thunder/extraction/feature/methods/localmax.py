from thunder.extraction.feature.base import FeatureMethod, FeatureAlgorithm


class LocalMax(FeatureMethod):

    def __init__(self, **kwargs):
        algorithm = LocalMaxAlgorithm(**kwargs)
        super(self.__class__, self).__init__(algorithm, **kwargs)


class LocalMaxAlgorithm(FeatureAlgorithm):

    def __init__(self, threshold=10, **extra):
        self.threshold = threshold

    def extract(self, data):
        return data.mean()

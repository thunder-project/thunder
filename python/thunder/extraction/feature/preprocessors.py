from thunder.extraction.feature.base import FeaturePreprocessor


class MeanPreprocessor(FeaturePreprocessor):

    def preprocess(self, data):
        return data.mean()

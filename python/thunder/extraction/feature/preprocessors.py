from thunder.rdds.images import Images
from thunder.rdds.series import Series
from thunder.extraction.feature.base import FeaturePreprocessor


class MeanPreprocessor(FeaturePreprocessor):

    def preprocess(self, data):
        if isinstance(data, Images):
            return data.mean()
        elif isinstance(data, Series):
            return data.seriesMean().pack()

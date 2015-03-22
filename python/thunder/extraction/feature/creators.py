from thunder.rdds.images import Images
from thunder.rdds.series import Series
from thunder.extraction.feature.base import FeatureCreator


class MeanFeatureCreator(FeatureCreator):
    """
    Compute the mean across images for every pixel
    """
    def create(self, data):
        if isinstance(data, Images):
            return data.mean()
        elif isinstance(data, Series):
            return data.seriesMean().pack()


class StdevFeatureCreator(FeatureCreator):
    """
    Compute the standard deviation across images for every pixel
    """
    def create(self, data):
        if isinstance(data, Images):
            return data.stdev()
        elif isinstance(data, Series):
            return data.seriesStdev().pack()

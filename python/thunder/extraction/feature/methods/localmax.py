from thunder.extraction.feature.base import FeatureMethod, FeatureAlgorithm
from thunder.extraction.feature.preprocessors import MeanPreprocessor
from thunder.extraction.source import SourceModel, Source


class LocalMax(FeatureMethod):

    def __init__(self, **kwargs):
        algorithm = LocalMaxAlgorithm(**kwargs)
        preprocess = MeanPreprocessor()
        super(self.__class__, self).__init__(algorithm, preprocess, **kwargs)


class LocalMaxAlgorithm(FeatureAlgorithm):

    def __init__(self, min_distance=10, num_peaks=None, **extra):
        self.min_distance = min_distance
        self.num_peaks = num_peaks

    def extract(self, im):
        from numpy import ones, concatenate
        from skimage.feature import peak_local_max

        if im.ndim == 2:
            peaks = peak_local_max(im, min_distance=self.min_distance, num_peaks=self.num_peaks).tolist()
        else:
            peaks = []
            for i in range(0, im.shape[2]):
                tmp = peak_local_max(im[:, :, i], min_distance=self.min_distance, num_peaks=self.num_peaks)
                peaks = peaks.append(concatenate((tmp, ones((len(tmp), 1)) * i), axis=1))

        return SourceModel([Source([p]) for p in peaks])
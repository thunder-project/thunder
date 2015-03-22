from numpy import cos, sin, pi

from thunder.extraction.feature.base import FeatureMethod, FeatureAlgorithm
from thunder.extraction.feature.preprocessors import MeanPreprocessor
from thunder.extraction.source import SourceModel, Source


class LocalMax(FeatureMethod):

    def __init__(self, **kwargs):
        algorithm = LocalMaxAlgorithm(**kwargs)
        preprocess = MeanPreprocessor()
        super(self.__class__, self).__init__(algorithm, preprocess, **kwargs)


class LocalMaxAlgorithm(FeatureAlgorithm):

    def __init__(self, minDistance=10, numPeaks=None, **extra):
        self.minDistance = minDistance
        self.numPeaks = numPeaks

    def extract(self, im, radius=5, res=10):
        from numpy import ones, concatenate
        from skimage.feature import peak_local_max

        if im.ndim == 2:
            peaks = peak_local_max(im, min_distance=self.minDistance, num_peaks=self.numPeaks).tolist()
        else:
            peaks = []
            for i in range(0, im.shape[2]):
                tmp = peak_local_max(im[:, :, i], min_distance=self.minDistance, num_peaks=self.numPeaks)
                peaks = peaks.append(concatenate((tmp, ones((len(tmp), 1)) * i), axis=1))

        # convert row/col to x/y
        peaks = map(lambda p: p[::-1], peaks)
        def pointToCircle(center):
            xy = [(cos(2 * pi/res * x) * radius + center[0], sin(2 * pi/res * x) * radius + center[1])
                  for x in xrange(0, res+1)]
            return xy

        return SourceModel([Source(pointToCircle(p)) for p in peaks])
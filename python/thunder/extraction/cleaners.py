from thunder.extraction.source import SourceModel


class Cleaner(object):

    def clean(self, sources):
        raise NotImplementedError


class BasicCleaner(Cleaner):
    """
    A simple cleaner that just removes sources larger or smaller than specified sizes.
    """
    def __init__(self, minArea=0, maxArea=200, **extra):
        self.minArea = minArea
        self.maxArea = maxArea

    def clean(self, model):

        if not isinstance(model, SourceModel):
            raise Exception("Input must be Source Model, got %s" % type(model))

        crit = lambda s: (s.area > self.minArea) and (s.area < self.maxArea)
        new = filter(crit, model.sources)

        if len(new) < 1:
            raise Exception("Filtering removed all sources, try different parameters?")

        return SourceModel(new)
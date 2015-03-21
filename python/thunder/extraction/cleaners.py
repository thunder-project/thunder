from thunder.extraction.source import SourceModel

class Cleaner(object):

    def clean(self, sources):
        raise NotImplementedError


class BasicCleaner(Cleaner):
    """
    A simple cleaner that removes sources larger than a specified size.
    """
    def __init__(self, minSize=50, **extra):
        self.minSize = minSize

    def clean(self, model):

        if not isinstance(model, SourceModel):
            raise Exception("Input must be Source Model, got %s" % type(model))

        new = filter(lambda s: len(s.coordinates) > self.minSize, model.sources)
        return SourceModel(new)
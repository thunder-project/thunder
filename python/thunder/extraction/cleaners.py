class Cleaner(object):

    def clean(self, sources):
        raise NotImplementedError


class BasicCleaner(Cleaner):

    def __init__(self, minSize=10, **extra):
        self.minSize = minSize

    def clean(self, sources):
        return filter(lambda x: len(x.coordinates) > self.minSize, sources)
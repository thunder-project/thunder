
class Cleaner(object):

    def clean(self, sources, data=None):
        raise NotImplementedError


class BasicCleaner(Cleaner):

    def __init__(self, threshold=1, **extra):
        self.threshold = threshold

    def clean(self, sources, data=None):
        return sources
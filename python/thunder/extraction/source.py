from thunder.utils.serializable import Serializable
from thunder.rdds.images import Images
from thunder.rdds.series import Series


class Source(Serializable, object):

    def __init__(self, coordinates, values=None, id=None, bbox=None):
        self.coordinates = coordinates
        if values is not None:
            self.values = values
        if id is not None:
            self.id = id
        if bbox is not None:
            self.bbox = bbox

    def __repr__(self):
        s = self.__class__.__name__
        s += '\ncoordinates: %s' % (repr(self.coordinates))
        for opt in ["values", "id", "bbox"]:
            if hasattr(self, opt):
                s += '\n%s: %s' % (opt, repr(self.__getattribute__(opt)))
        return s


class SourceModel(Serializable, object):

    def __init__(self, sources):
        self.sources = sources

    def transform(self, data):

        if not (isinstance(data, Images) or isinstance(data, Series)):
            raise Exception("Input must either be Images or Series (or a subclass)")

    def save(self, f, numpyStorage='auto', **kwargs):
        """
        Custom save with simplier output
        """
        simplify = lambda d: d['sources']['py/homogeneousList']['data']
        super(SourceModel, self).save(f, numpyStorage, simplify=simplify, **kwargs)

    @classmethod
    def load(cls, f, numpyStorage='auto', **kwargs):
        """
        Custom load to handle simplified output
        """
        unsimplify = lambda d: {'sources': {
            'py/homogeneousList': {'data': d, 'module': 'thunder.extraction.source', 'type': 'Source'}}}
        return super(SourceModel, cls).load(f, unsimplify=unsimplify)

    def __repr__(self):
        s = self.__class__.__name__
        s += '\n%g sources' % (len(self.sources))
        return s
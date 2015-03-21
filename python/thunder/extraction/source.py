from numpy import asarray, median, sqrt, ndarray, amin, amax, concatenate

from thunder.utils.serializable import Serializable
from thunder.rdds.images import Images
from thunder.rdds.series import Series


class Source(Serializable, object):
    """
    A single source, represented as a list of coordinates and other optional specifications.

    Parameters
    ----------
    coordinates : array-like
        List of 2D or 3D coordinates, can be a list of lists or array of shape (n,2) or (n,3)

    values : list or array-like
        Value (or weight) associated with each coordiante

    id : int or string
        Arbitrary specification per source, typically an index or string label

    bbox : list or array-like
        Boundaries of the source (with the lowest values for all axes followed by the highest values)

    center : list or array-like
        The coordinates of the center of the source
    """
    def __init__(self, coordinates, values=None, id=None, center=None, bbox=None):
        self.coordinates = asarray(coordinates)

        if self.coordinates.ndim == 1:
            self.coordinates = asarray([self.coordinates])

        if values is not None:
            self.values = asarray(values)
            if self.values.ndim == 0:
                self.values = asarray([self.values])

        if id is not None:
            self.id = id
        if center is None:
            self.center = self.findCenter()
        if bbox is None:
            self.bbox = self.findBox()

    def findCenter(self):
        return median(self.coordinates, axis=0)

    def findBox(self):
        mn = amin(self.coordinates, axis=0)
        mx = amax(self.coordinates, axis=0)
        return concatenate((mn, mx))

    def compareCenters(self, other):
        return sqrt(sum((self.center - other.center) ** 2))

    def tolist(self):
        """
        Convert all array-like attributes to list
        """
        import copy
        new = copy.copy(self)
        for prop in ["coordinates", "values", "center", "bbox"]:
            if hasattr(new, prop):
                val = getattr(new, prop)
                if val is not None and not isinstance(val, list):
                    setattr(new, prop, val.tolist())
        return new

    def toarray(self):
        """
        Convert all array-like attributes to ndarray
        """
        import copy
        new = copy.copy(self)
        for prop in ["coordinates", "values", "center", "bbox"]:
            if hasattr(new, prop):
                val = new.__getattribute__(prop)
                if val is not None and not isinstance(val, ndarray):
                    setattr(new, prop, asarray(val))
        return new

    def __repr__(self):
        s = self.__class__.__name__
        c = self.coordinates
        cs = c.tolist() if isinstance(c, ndarray) else c
        s += '\ncoordinates: %s' % (repr(cs))
        for opt in ["values", "id", "center", "bbox"]:
            if hasattr(self, opt):
                o = self.__getattribute__(opt)
                os = o.tolist() if isinstance(o, ndarray) else o
                s += '\n%s: %s' % (opt, repr(os))
        return s


class SourceModel(Serializable, object):
    """
    A source model as a collection of extracted sources.

    Parameters
    ----------
    sources : list or Sources or a single Source
        The identified sources

    See also
    --------
    Source
    """
    def __init__(self, sources):
        if isinstance(sources, Source):
            self.sources = [sources]
        elif isinstance(sources, list) and isinstance(sources[0], Source):
            self.sources = sources
        else:
            raise Exception("Input type not recognized, must be Source or list of Sources, got %s" % type(sources))

    def __getitem__(self, entry):
        if not isinstance(entry, int):
            raise IndexError("Selection not recognized, must be Int, got %s" % type(entry))
        return self.sources[entry]

    @property
    def coordinates(self):
        all = []
        for s in self.sources:
            all.append(s.coordinates.tolist())
        return asarray(all)

    @property
    def centers(self):
        all = []
        for s in self.sources:
            all.append(s.center.tolist())
        return asarray(all)

    def transform(self, data, collect=True):
        """
        Extract time series from data using a list of sources.

        Currently only supports simple averaging over coordinates.
        TODO add support for weighting

        Params
        ------
        data : Images or Series object
            The data to extract

        collect : boolean, optional, default = True
            Whether to collect to local array or keep as a Series
        """
        if not (isinstance(data, Images) or isinstance(data, Series)):
            raise Exception("Input must either be Images or Series (or a subclass)")

        if isinstance(data, Images):
            output = data.meanByRegions(self.coordinates).toSeries()
        else:
            output = data.meanByRegions(self.coordinates)

        if collect:
            return output.collectValuesAsArray()
        else:
            return output

    def save(self, f, numpyStorage='auto', **kwargs):
        """
        Custom save with simpler, more human-readable output
        """
        self.sources = map(lambda s: s.tolist(), self.sources)
        simplify = lambda d: d['sources']['py/homogeneousList']['data']
        super(SourceModel, self).save(f, numpyStorage='ascii', simplify=simplify, **kwargs)

    @classmethod
    def load(cls, f, numpyStorage='auto', **kwargs):
        """
        Custom load to handle simplified, more human-readable output
        """
        unsimplify = lambda d: {'sources': {
            'py/homogeneousList': {'data': d, 'module': 'thunder.extraction.source', 'type': 'Source'}}}
        output = super(SourceModel, cls).load(f, unsimplify=unsimplify)
        output.sources = map(lambda s: s.toarray(), output.sources)
        return output

    def __repr__(self):
        s = self.__class__.__name__
        s += '\n%g sources' % (len(self.sources))
        return s
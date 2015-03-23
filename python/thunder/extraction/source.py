from numpy import asarray, mean, sqrt, ndarray, amin, amax, concatenate, sum, zeros, maximum

from thunder.utils.serializable import Serializable
from thunder.rdds.images import Images
from thunder.rdds.series import Series


class Source(Serializable, object):
    """
    A single source, represented as a list of coordinates and other optional specifications.

    A source also has a set of lazily computed attributes useful for representing and comparing
    its geometry, such as center, bounding box, and bounding polygon. These properties
    will be computed lazily and made available as attributes when requested.

    Parameters
    ----------
    coordinates : array-like
        List of 2D or 3D coordinates, can be a list of lists or array of shape (n,2) or (n,3)

    values : list or array-like
        Value (or weight) associated with each coordiante

    id : int or string
        Arbitrary specification per source, typically an index or string label

    Attributes
    ----------
    center : list or array-like
        The coordinates of the center of the source

    polygon : list or array-like
        The coordinates of a polygon bounding the region (a convex hull)

    bbox : list or array-like
        Boundaries of the source (with the lowest values for all axes followed by the highest values)

    area : scalar
        The area of the region
    """
    from zope.cachedescriptors import property

    def __init__(self, coordinates, values=None, id=None):
        self.coordinates = asarray(coordinates)

        if self.coordinates.ndim == 1:
            self.coordinates = asarray([self.coordinates])

        if values is not None:
            self.values = asarray(values)
            if self.values.ndim == 0:
                self.values = asarray([self.values])

        if id is not None:
            self.id = id

    @property.Lazy
    def center(self):
        """
        Find the region center using a mean.
        """
        # TODO Add option to use weights
        return mean(self.coordinates, axis=0)

    @property.Lazy
    def polygon(self):
        """
        Find the bounding polygon as a convex hull
        """
        # TODO Add option for simplification
        from scipy.spatial import ConvexHull
        if len(self.coordinates) >= 4:
            inds = ConvexHull(self.coordinates).vertices
            return self.coordinates[inds]
        else:
            return self.coordinates

    @property.Lazy
    def bbox(self):
        """
        Find the bounding box.
        """
        mn = amin(self.coordinates, axis=0)
        mx = amax(self.coordinates, axis=0)
        return concatenate((mn, mx))

    @property.Lazy
    def area(self):
        """
        Find the region area.
        """
        return len(self.coordinates)

    def restore(self, skip=None):
        """
        Remove all lazy properties, will force recomputation
        """
        if skip is None:
            skip = []
        elif isinstance(skip, str):
            skip = [skip]
        for prop in LAZY_ATTRIBUTES:
            if prop in self.__dict__.keys() and prop not in skip:
                del self.__dict__[prop]
        return self

    def distance(self, other):
        """
        Distance between the center of this source and another.
        """
        if isinstance(other, Source):
            return sqrt(sum((self.center - other.center) ** 2))
        elif isinstance(other, list) or isinstance(other, ndarray):
            return sqrt(sum((self.center - asarray(other)) ** 2))

    def tolist(self):
        """
        Convert array-like attributes to list
        """
        import copy
        new = copy.copy(self)
        for prop in ["coordinates", "values", "center", "bbox", "polygon"]:
            if prop in self.__dict__.keys():
                val = new.__getattribute__(prop)
                if val is not None and not isinstance(val, list):
                    setattr(new, prop, val.tolist())
        return new

    def toarray(self):
        """
        Convert array-like attributes to ndarray
        """
        import copy
        new = copy.copy(self)
        for prop in ["coordinates", "values", "center", "bbox", "polygon"]:
            if prop in self.__dict__.keys():
                val = new.__getattribute__(prop)
                if val is not None and not isinstance(val, ndarray):
                    setattr(new, prop, asarray(val))
        return new

    def mask(self, dims=None, binary=True):
        """
        Construct a mask from a source, either locally or within a larger image.

        Parameters
        ----------
        dims : list or tuple, optional, default = None
            Dimensions of large image in which to draw mask. If none, will restrict
            to the bounding box of the region.

        binary : boolean, optional, deafult = True
            Whether to incoporate values or only show a binary mask
        """
        if dims is None:
            extent = self.bbox[len(self.center):] - self.bbox[0:len(self.center)] + 1
            empty = zeros(extent)
            coords = (self.coordinates - self.bbox[0:len(self.center)])
        else:
            empty = zeros(dims)
            coords = self.coordinates

        if hasattr(self, 'values') and self.values is not None and binary is False:
            empty[coords.T.tolist()] = self.values
        else:
            empty[coords.T.tolist()] = 1

        return empty

    def __repr__(self):
        s = self.__class__.__name__
        for opt in ["id", "center", "bbox"]:
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

    def combiner(self, prop, tolist=True):
        combined = []
        for s in self.sources:
            p = getattr(s, prop)
            if tolist:
                p = p.tolist()
            combined.append(p)
        return combined

    @property
    def coordinates(self):
        """
        List of coordinates combined across sources
        """
        return self.combiner('coordinates')

    @property
    def values(self):
        """
        List of coordinates combined across sources
        """
        return self.combiner('values')

    @property
    def centers(self):
        """
        Array of centers combined across sources
        """
        return asarray(self.combiner('center'))

    @property
    def polygons(self):
        """
        List of polygons combined across sources
        """
        return self.combiner('polygon')

    @property
    def areas(self):
        """
        List of areas combined across sources
        """
        return self.combiner('area', tolist=False)

    def masks(self, dims=None, binary=True):
        """
        Composite mask combined across sources
        """
        if dims is None:
            raise Exception("Must provide image dimensions for composite masks.")

        combined = zeros(dims)
        for s in self.sources:
            combined = maximum(s.mask(dims, binary), combined)
        return combined

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

        # inversion converts x/y to row/col
        if isinstance(data, Images):
            output = data.meanByRegions(self.coordinates).toSeries()
        else:
            output = data.meanByRegions(self.coordinates)

        if collect:
            return output.collectValuesAsArray()
        else:
            return output

    def save(self, f, numpyStorage='auto', include=None, **kwargs):
        """
        Custom save with simplified, human-readable output, and selection of lazy attributes.
        """
        import copy
        output = copy.deepcopy(self)
        if isinstance(include, str):
            include = [include]
        if include is not None:
            for prop in include:
                map(lambda s: getattr(s, prop), output.sources)
        output.sources = map(lambda s: s.restore(include).tolist(), output.sources)
        simplify = lambda d: d['sources']['py/homogeneousList']['data']
        super(SourceModel, output).save(f, numpyStorage='ascii', simplify=simplify, **kwargs)

    @classmethod
    def load(cls, f, **kwargs):
        """
        Custom load to handle simplified, human-readable output
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

LAZY_ATTRIBUTES = ["center", "polygon", "bbox", "area"]
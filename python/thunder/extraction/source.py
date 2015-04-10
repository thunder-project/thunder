from numpy import asarray, mean, sqrt, ndarray, amin, amax, concatenate, sum, zeros, maximum, \
    argmin, newaxis, ones, delete, NaN, inf, isnan

from thunder.utils.serializable import Serializable
from thunder.utils.common import checkParams
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

    def transform(self, data, collect=True):
        """
        Extract series from data using a list of sources.

        Currently only supports averaging over coordinates.

        Params
        ------
        data : Images or Series object
            The data from which to extract

        collect : boolean, optional, default = True
            Whether to collect to local array or keep as a Series
        """

        if not (isinstance(data, Images) or isinstance(data, Series)):
            raise Exception("Input must either be Images or Series (or a subclass)")

        # TODO add support for weighting
        if isinstance(data, Images):
            output = data.meanByRegions([self.coordinates]).toSeries()
        else:
            output = data.meanOfRegion(self.coordinates)

        if collect:
            return output.collectValuesAsArray()
        else:
            return output

    def mask(self, dims=None, binary=True, outline=False):
        """
        Construct a mask from a source, either locally or within a larger image.

        Parameters
        ----------
        dims : list or tuple, optional, default = None
            Dimensions of large image in which to draw mask. If none, will restrict
            to the bounding box of the region.

        binary : boolean, optional, deafult = True
            Whether to incoporate values or only show a binary mask

        outline : boolean, optional, deafult = False
            Whether to only show outlines (derived using binary dilation)
        """
        coords = self.coordinates

        if dims is None:
            extent = self.bbox[len(self.center):] - self.bbox[0:len(self.center)] + 1
            m = zeros(extent)
            coords = (coords - self.bbox[0:len(self.center)])
        else:
            m = zeros(dims)

        if hasattr(self, 'values') and self.values is not None and binary is False:
            m[coords.T.tolist()] = self.values
        else:
            m[coords.T.tolist()] = 1

        if outline:
            from skimage.morphology import binary_dilation
            m = binary_dilation(m, ones((3, 3))) - m

        return m

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
        elif isinstance(sources, list):
            self.sources = []
            for ss in sources:
                self.sources.append(Source(ss))
        else:
            raise Exception("Input type not recognized, must be Source, list of Sources, "
                            "or list of coordinates, got %s" % type(sources))

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

    def masks(self, dims=None, binary=True, outline=False, base=None):
        """
        Composite masks combined across sources as an iamge.

        Parameters
        ----------
        dims : list or tuple, optional, default = None
            Dimensions of image in which to create masks, must either provide
            these or provide a base image

        binary : boolean, optional, deafult = True
            Whether to incoporate values or only show a binary mask

        outline : boolean, optional, deafult = False
            Whether to only show outlines (derived using binary dilation)

        base : array-like, optional, deafult = None
            Base background image on which to put masks.
        """
        from thunder import Colorize

        if dims is None and base is None:
            raise Exception("Must provide image dimensions for composite masks "
                            "or provide a base image.")

        if base is not None and isinstance(base, SourceModel):
            outline = True

        if dims is None and base is not None:
            dims = asarray(base).shape

        combined = zeros(dims)
        for s in self.sources:
            combined = maximum(s.mask(dims, binary, outline), combined)

        if base is not None:
            if isinstance(base, SourceModel):
                base = base.masks(dims)
                baseColor = 'silver'
            else:
                baseColor = 'white'
            clr = Colorize(cmap='indexed', colors=[baseColor, 'deeppink'])
            combined = clr.transform([base, combined])

        return combined

    def match(self, other, unique=False, minDistance=inf):
        """
        For each source in self, find the index of the closest source in other.

        Uses euclidean distances between centers to determine distances.

        Can select nearest matches with or without enforcing uniqueness;
        if unique is False, will return the closest source in other for
        each source in self, possibly repeating sources multiple times
        if unique is True, will only allow each source in other to be matched
        with a single source in self, as determined by a greedy selection procedure.
        The minDistance parameter can be used to prevent far-away sources from being
        chosen during greedy selection.

        Params
        ------
        other : SourceModel
            The source model to match sources to

        unique : boolean, optional, deafult = True
            Whether to only return unique matches

        minDistance : scalar, optiona, default = inf
            Minimum distance to use when selecting matches
        """
        from scipy.spatial.distance import cdist

        targets = other.centers
        targetInds = range(0, len(targets))
        matches = []
        for s in self.sources:
            update = 1

            # skip if no targets left, otherwise update
            if len(targets) == 0:
                update = 0
            else:
                dists = cdist(targets, s.center[newaxis])
                if dists.min() < minDistance:
                    ind = argmin(dists)
                else:
                    update = 0

            # apply updates, otherwise add a nan
            if update == 1:
                matches.append(targetInds[ind])
                if unique is True:
                    targets = delete(targets, ind, axis=0)
                    targetInds = delete(targetInds, ind)
            else:
                matches.append(NaN)

        return matches

    def distance(self, other, minDistance=inf):
        """
        Compute the distance between each source in self and other.

        First estimates a matching source from other for each source
        in self, then computes the distance between the two sources.
        The matches are unique, using a greedy procedure,
        and minDistance can be used to prevent outliers during matching.

        Parameters
        ----------
        other : SourceModel
            The sources to compute distances to

        minDistance : scalar, optiona, default = inf
            Minimum distance to use when selecting matches
        """

        inds = self.match(other, unique=True, minDistance=minDistance)
        d = []
        for jj, ii in enumerate(inds):
            if ii is not NaN:
                d.append(self[jj].distance(other[ii]))
            else:
                d.append(NaN)
        return asarray(d)

    def similarity(self, other, metric='distance', thresh=5):
        """
        Estimate similarity between sources in self and other.

        Will compute the fraction of sources in self that are found
        in other, based on a given distance metric and a threshold.
        The fraction is estimated as the number of sources in self
        found in other, divided by the total number of sources in self.

        Parameters
        ----------
        other : SourceModel
            The sources to compare to

        metric : str, optional, default = "distance"
            Metric to use when computing distances

        thresh : scalar, optional, default = 5
            The distance below which a source is considered found
        """

        checkParams(metric, ['distance'])

        if metric == 'distance':
            vals = self.distance(other, minDistance=thresh)
            vals[isnan(vals)] = inf
        else:
            raise Exception("Metric not recognized")

        hits = sum(vals < thresh) / float(len(self.sources))

        return hits

    def transform(self, data, collect=True):
        """
        Extract series from data using a list of sources.

        Currently only supports simple averaging over coordinates.

        Params
        ------
        data : Images or Series object
            The data from which to extract signals

        collect : boolean, optional, default = True
            Whether to collect to local array or keep as a Series
        """
        if not (isinstance(data, Images) or isinstance(data, Series)):
            raise Exception("Input must either be Images or Series (or a subclass)")

        # TODO add support for weighting
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
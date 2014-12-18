from numpy import corrcoef, clip

from thunder.rdds.series import Series


class SpatialSeries(Series):
    """
    Distributed collection of 1d array data with spatial coordinates.

    Backed by an RDD of key-value pairs where the
    key is a tuple identifying a spatial coordinate (e.g. x,y,z),
    and the value is a one-dimensional array.

    Parameters
    ----------
    rdd : RDD of (tuple, array) pairs
        RDD containing the series data

    index : array-like or one-dimensional list
        Values must be unique, same length as the arrays in the input data.
        Defaults to arange(len(data)) if not provided.

    dims : Dimensions
        Specify the dimensions of the spatial coordinate keys (min, max, and count),
        can avoid computation if known in advance

    See also
    --------
    Series : base class for Series data
    """
    # use superclass __init__

    @property
    def _constructor(self):
        return SpatialSeries

    def mapToNeighborhood(self, neighborhood):
        """
        Flat map records to key-value pairs where the
        key is neighborhood identifier
        """

        def toNeighbors(ind, v, sz, mn, mx):
            """Create a list of key value pairs with multiple shifted copies
            of each record over a region specified by sz
            """
            rngX = range(-sz, sz+1, 1)
            rngY = range(-sz, sz+1, 1)
            out = list()
            for x in rngX:
                for y in rngY:
                    newX = clip(ind[0] + x, mn[0], mx[0])
                    newY = clip(ind[1] + y, mn[1], mx[1])
                    newInd = (newX, newY, ind[2])
                    out.append((newInd, v))
            return out
        dims = self.dims
        rdd = self.rdd.flatMap(lambda (k, v): toNeighbors(k, v, neighborhood, dims.min[0:2], dims.max[0:2]))

        return self._constructor(rdd).__finalize__(self)

    def localCorr(self, neighborhood):

        if len(self.dims.max) not in [2, 3]:
                raise NotImplementedError('keys must have 2 or 3 dimensions to compute local correlations')

        # flat map to key value pairs where the key is neighborhood identifier and value is time series
        neighbors = self.mapToNeighborhood(neighborhood)

        # reduce by key to get the average time series for each neighborhood
        means = neighbors.rdd.reduceByKey(lambda x, y: x + y).mapValues(lambda x: x / ((2*neighborhood+1)**2))

        # join with the original time series data to compute correlations
        result = self.rdd.join(means)

        # get correlations
        corr = result.mapValues(lambda x: corrcoef(x[0], x[1])[0, 1])

        # force sorting, but reverse keys for correct ordering
        output = corr.map(lambda (k, v): (k[::-1], v)).sortByKey().map(lambda (k, v): (k[::-1], v))
        return Series(output, index='correlation').__finalize__(self)
from ..base import Base


class Blocks(Base):
    """
    Superclass for subdivisions of Images data.

    Subclasses of Blocks will be returned by an images.toBlocks() call.
    """
    _metadata = Base._metadata + ['blockshape']

    def __init__(self, values):
        super(Blocks, self).__init__(values)

    @property
    def _constructor(self):
        return Blocks

    @property
    def blockshape(self):
        return tuple(self.values.plan)

    def count(self):
        """
        Explicit count of the number of items.

        For lazy or distributed data, will force a computation.
        """
        return self.tordd().count()

    def map(self, func, dims=None):
        """
        Apply an array -> array function to each block
        """
        mapped = self.values.map(func, value_shape=dims)
        return self._constructor(mapped).__finalize__(self)

    def first(self):
        """
        Return the first element.
        """
        return self.values.tordd().values().first()

    def toimages(self):
        """
        Convert blocks to images.
        """
        from thunder.images.images import Images
        values = self.values.values_to_keys((0,)).unchunk()
        return Images(values)

    def toseries(self):
        """
        Converts blocks to series.
        """
        from thunder.series.series import Series
        values = self.values.values_to_keys(tuple(range(1, len(self.shape)))).unchunk()
        return Series(values)

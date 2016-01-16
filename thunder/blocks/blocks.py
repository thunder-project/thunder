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
    def subshape(self):
        return tuple(self.values.plan)

    def count(self):
        """
        Explicit count of the number of items.

        For lazy or distributed data, will force a computation.
        """
        return self.tordd().count()

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
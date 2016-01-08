from numpy import prod, unravel_index
import cStringIO as StringIO
import struct

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
        from thunder.images.readers import frombolt
        values = self.values.values_to_keys((0,)).unchunk()
        return frombolt(values)

    def toseries(self):
        """
        Converts blocks to series.
        """
        from thunder.series.series import Series
        values = self.values.values_to_keys(tuple(range(1, len(self.shape)))).unchunk()
        return Series(values)

    def getbinary(self):
        """
        Extract chunks of binary data for each block.

        The keys of each chunk should be filenames ending in ".bin".
        The values should be packed binary data.

        Subclasses that can be converted to a Series object are expected to override this method.
        """
        from thunder.series.writers import getlabel

        def getblock(kv):
            key, val = kv
            dims = val.shape[1:]
            label = getlabel(key[1])+".bin"
            packer = None
            buf = StringIO.StringIO()
            for i in range(prod(dims)):
                ind = unravel_index(i, dims)
                series = val[(slice(None, None),) + ind]
                if packer is None:
                    packer = struct.Struct('h'*len(ind))
                buf.write(packer.pack(*ind))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return label, val

        return self.tordd().map(getblock)

    def tobinary(self, path, overwrite=False, credentials=None):
        """
        Writes out Series-formatted binary data.

        Subclasses are *not* expected to override this method.

        Parameters
        ----------
        path : string
            Output files will be written underneath path.
            This directory must not yet exist (unless overwrite is True),
            and must be no more than one level beneath an existing directory.
            It will be created as a result of this call.

        overwrite : bool
            If true, outputdirname and all its contents will
            be deleted and recreated as part of this call.
        """
        from thunder.writers import get_parallel_writer
        from thunder.series.writers import write_config

        if not overwrite:
            from thunder.utils import check_path
            check_path(path, credentials=credentials)
            overwrite = True

        writer = get_parallel_writer(path)(path, overwrite=overwrite, credentials=credentials)
        binary = self.getbinary()
        binary.foreach(writer.write)
        write_config(path, len(self.shape) - 1, self.shape[0],
                     keytype='int16', valuetype=self.dtype, overwrite=overwrite)
from thunder import credentials

def tobinary(series, path, overwrite=False):
    """
    Writes out data to binary format.

    Parameters
    ----------
    series : Series
        The data to write

    path : string path or URI to directory to be created
        Output files will be written underneath path.
        Directory will be created as a result of this call.

    overwrite : bool
        If true, path and all its contents will be deleted and
        recreated as partof this call.
    """
    import cStringIO as StringIO
    import struct
    from thunder import credentials
    from thunder.utils.common import check_path
    from thunder.data.fileio.writers import getParallelWriterForPath

    if not overwrite:
        check_path(path, credentials=credentials())
        overwrite = True  # prevent additional downstream checks for this path

    def getbinary(kvIter):
        """ Collects all Series records in a partition into a single binary series record. """
        keypacker = None
        firstKey = None
        buf = StringIO.StringIO()
        for seriesKey, series in kvIter:
            if keypacker is None:
                keypacker = struct.Struct('h'*len(seriesKey))
                firstKey = seriesKey
            # print >> sys.stderr, seriesKey, series, series.tostring().encode('hex')
            buf.write(keypacker.pack(*seriesKey))
            buf.write(series.tostring())
        val = buf.getvalue()
        buf.close()
        # we might have an empty partition, in which case firstKey will still be None
        if firstKey is None:
            return iter([])
        else:
            label = getlabel(firstKey) + ".bin"
            return iter([(label, val)])

    writer = getParallelWriterForPath(path)(path, overwrite=overwrite, credentials=credentials())

    binary = series.rdd.mapPartitions(getbinary)

    binary.foreach(writer.writerFcn)

    firstKey, firstVal = series.first()
    write_config(path, len(firstKey), len(firstVal), keytype='int16',
                 valuetype=series.dtype, overwrite=overwrite)

def write_config(path, nkeys, nvalues, keytype='int16', valuetype='int16',
                 name="conf.json", overwrite=True):
    """
    Helper function to write out a conf.json file with required information to load Series binary data.
    """
    import json
    from thunder.data.fileio.writers import getFileWriterForPath

    writer = getFileWriterForPath(path)
    conf = {'input': path, 'nkeys': nkeys, 'nvalues': nvalues,
            'valuetype': str(valuetype), 'keytype': str(keytype)}

    confwriter = writer(path, name, overwrite=overwrite, credentials=credentials())
    confwriter.writeFile(json.dumps(conf, indent=2))

    successwriter = writer(path, "SUCCESS", overwrite=overwrite, credentials=credentials())
    successwriter.writeFile('')

def getlabel(key):
    return '-'.join(reversed(["key%02d_%05g" % (ki, k) for (ki, k) in enumerate(key)]))
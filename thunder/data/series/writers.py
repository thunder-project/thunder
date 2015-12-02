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
    from thunder.data.writers import get_parallel_writer

    if not overwrite:
        check_path(path, credentials=credentials())
        overwrite = True

    def tobuffer(kv):
        keypacker = None
        firstkey = None
        buf = StringIO.StringIO()
        for serieskey, series in kv:
            if keypacker is None:
                keypacker = struct.Struct('h'*len(serieskey))
                firstkey = serieskey
            buf.write(keypacker.pack(*serieskey))
            buf.write(series.tostring())
        val = buf.getvalue()
        buf.close()
        if firstkey is None:
            # empty partition
            return iter([])
        else:
            label = getlabel(firstkey) + ".bin"
            return iter([(label, val)])

    writer = get_parallel_writer(path)(path, overwrite=overwrite, credentials=credentials())
    binary = series.rdd.mapPartitions(tobuffer)
    binary.foreach(writer.write)
    firstkey, firstvalue = series.first()
    write_config(path, len(firstkey), len(firstvalue), keytype='int16',
                 valuetype=series.dtype, overwrite=overwrite)

def write_config(path, nkeys, nvalues, keytype='int16', valuetype='int16',
                 name="conf.json", overwrite=True):
    """
    Write a conf.json file with required information to load Series binary data.
    """
    import json
    from thunder.data.writers import get_file_writer

    writer = get_file_writer(path)
    conf = {'input': path, 'nkeys': nkeys, 'nvalues': nvalues,
            'valuetype': str(valuetype), 'keytype': str(keytype)}

    confwriter = writer(path, name, overwrite=overwrite, credentials=credentials())
    confwriter.write(json.dumps(conf, indent=2))

    successwriter = writer(path, "SUCCESS", overwrite=overwrite, credentials=credentials())
    successwriter.write('')

def getlabel(key):
    """
    Get a file label from keys with reversed order
    """
    return '-'.join(reversed(["key%02d_%05g" % (ki, k) for (ki, k) in enumerate(key)]))
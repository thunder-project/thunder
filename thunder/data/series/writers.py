from thunder import credentials

def toBinary(series, outputDirPath, overwrite=False):
    """
    Writes out Series-formatted data.

    This method (Series.saveAsBinarySeries) writes out binary series files using the current partitioning
    of this Series object. (That is, if mySeries.rdd.getNumPartitions() == 5, then 5 files will be written
    out, one per partition.) The records will not be resorted; the file names for each partition will be
    taken from the key of the first Series record in that partition. If the Series object is already
    sorted and no records have been removed by filtering, then the resulting output should be equivalent
    to what one would get from calling myImages.saveAsBinarySeries().

    If all one wishes to do is to save out Images data in a binary series format, then
    tsc.convertImagesToSeries() will likely be more efficient than
    tsc.loadImages().toSeries().saveAsBinarySeries().

    Parameters
    ----------
    outputDirPath : string path or URI to directory to be created
        Output files will be written underneath outputdirname. This directory must not yet exist
        (unless overwrite is True), and must be no more than one level beneath an existing directory.
        It will be created as a result of this call.

    overwrite : bool
        If true, outputdirname and all its contents will be deleted and recreated as part
        of this call.
    """
    import cStringIO as StringIO
    import struct
    from thunder import credentials
    from thunder.data.blocks.blocks import SimpleBlocks
    from thunder.data.fileio.writers import getParallelWriterForPath

    if not overwrite:
        checkOverwrite(outputDirPath)
        overwrite = True  # prevent additional downstream checks for this path

    def partitionToBinarySeries(kvIter):
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
            label = SimpleBlocks.getBinarySeriesNameForKey(firstKey) + ".bin"
            return iter([(label, val)])

    writer = getParallelWriterForPath(outputDirPath)(outputDirPath, overwrite=overwrite, credentials=credentials())

    binseriesrdd = series.rdd.mapPartitions(partitionToBinarySeries)

    binseriesrdd.foreach(writer.writerFcn)

    # TODO: all we really need here are the number of keys and number of values, which could in principle
    # be cached in _nkeys and _nvals attributes, removing the need for this .first() call in most cases.
    firstKey, firstVal = series.first()
    writeSeriesConfig(outputDirPath, len(firstKey), len(firstVal), keyType='int16',
                      valueType=series.dtype, overwrite=overwrite)

def checkOverwrite(outputDirPath):
    from thunder.utils.common import raiseErrorIfPathExists
    raiseErrorIfPathExists(outputDirPath, credentials=credentials())

def writeSeriesConfig(outputDirPath, nkeys, nvalues, keyType='int16', valueType='int16',
                      confFilename="conf.json", overwrite=True):
    """
    Helper function to write out a conf.json file with required information to load Series binary data.
    """
    import json
    from thunder.data.fileio.writers import getFileWriterForPath

    filewriterClass = getFileWriterForPath(outputDirPath)
    # write configuration file
    # config JSON keys are lowercased "valuetype", "keytype", not valueType, keyType
    conf = {'input': outputDirPath,
            'nkeys': nkeys, 'nvalues': nvalues,
            'valuetype': str(valueType), 'keytype': str(keyType)}

    confWriter = filewriterClass(outputDirPath, confFilename, overwrite=overwrite, credentials=credentials())
    confWriter.writeFile(json.dumps(conf, indent=2))

    # touch "SUCCESS" file as final action
    successWriter = filewriterClass(outputDirPath, "SUCCESS", overwrite=overwrite, credentials=credentials())
    successWriter.writeFile('')

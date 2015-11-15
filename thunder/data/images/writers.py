from numpy import asarray

from thunder import credentials


def toPng(images, outputDirPath, cmap=None, vmin=None, vmax=None, prefix="image", overwrite=False):
    """
    Write out basic png files for two-dimensional image data.

    Files will be written into a newly-created directory given by outputdirname.

    Parameters
    ----------
    outputDirPath : string
        Path to output directory to be created. Exception will be thrown if this directory already
        exists, unless overwrite is True. Directory must be one level below an existing directory.

    prefix : string
        String to prepend to all filenames. Files will be named <prefix>-00000.png, <prefix>-00001.png, etc

    overwrite : bool
        If true, the directory given by outputdirname will first be deleted if it already exists.
    """
    dims = images.dims
    if not len(dims) == 2:
        raise ValueError("Only two-dimensional images can be exported as .png files; image is %d-dimensional." %
                         len(dims))

    from matplotlib.pyplot import imsave
    from io import BytesIO
    from thunder.data.fileio.writers import getParallelWriterForPath

    def toFilenameAndPngBuf(kv):
        key, img = kv
        fname = prefix+"-"+"%05d.png" % int(key)
        bytebuf = BytesIO()
        imsave(bytebuf, img, vmin, vmax, cmap=cmap, format="png")
        return fname, bytebuf.getvalue()

    bufRdd = images.rdd.map(toFilenameAndPngBuf)

    writer = getParallelWriterForPath(outputDirPath)(
        outputDirPath, overwrite=overwrite, credentials=credentials())
    bufRdd.foreach(writer.writerFcn)

def toBinary(images, outputDirPath, prefix="image", overwrite=False):
    """
    Write out images or volumes as flat binary files.

    Files will be written into a newly-created directory given by outputdirname.

    Parameters
    ----------
    outputDirPath : string
        Path to output directory to be created. Exception will be thrown if this directory already
        exists, unless overwrite is True. Directory must be one level below an existing directory.

    prefix : string
        String to prepend to all filenames. Files will be named <prefix>-00000.bin, <prefix>-00001.bin, etc

    overwrite : bool
        If true, the directory given by outputdirname will first be deleted if it already exists.
    """
    from thunder.data.fileio.writers import getParallelWriterForPath

    dimsTotal = list(asarray(images.dims.max)-asarray(images.dims.min)+1)

    def toFilenameAndBinaryBuf(kv):
        key, img = kv
        fname = prefix+"-"+"%05d.bin" % int(key)
        return fname, img.transpose().copy()

    bufRdd = images.rdd.map(toFilenameAndBinaryBuf)

    writer = getParallelWriterForPath(outputDirPath)(
        outputDirPath, overwrite=overwrite, credentials=credentials())
    bufRdd.foreach(writer.writerFcn)
    writeBinaryImagesConfig(
        outputDirPath, dims=dimsTotal, dtype=images.dtype, overwrite=overwrite)

def writeBinaryImagesConfig(outputDirPath, dims, dtype='int16', confFilename="conf.json", overwrite=True):
    """
    Helper function to write out a conf.json file with required information to load binary Image data.
    """
    import json
    from thunder.data.fileio.writers import getFileWriterForPath

    filewriterClass = getFileWriterForPath(outputDirPath)

    # write configuration file
    conf = {'dims': dims, 'dtype': dtype}
    confWriter = filewriterClass(
        outputDirPath, confFilename, overwrite=overwrite, credentials=credentials())
    confWriter.writeFile(json.dumps(conf, indent=2))

    # touch "SUCCESS" file as final action
    successWriter = filewriterClass(
        outputDirPath, "SUCCESS", overwrite=overwrite, credentials=credentials())
    successWriter.writeFile('')

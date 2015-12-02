from numpy import asarray
import json

from thunder import credentials


def topng(images, path, prefix="image", overwrite=False):
    """
    Write out PNG files for 2d or 3d image data.

    See also
    --------
    thunder.data.images.topng
    """
    dims = images.dims
    if not len(dims) in [2, 3]:
        raise ValueError("Only 2D or 3D images can be exported to png, "
                         "images are %d-dimensional." % len(dims))

    from scipy.misc import imsave
    from io import BytesIO
    from thunder.data.writers import get_parallel_writer

    def tobuffer(kv):
        key, img = kv
        fname = prefix+"-"+"%05d.png" % int(key)
        bytebuf = BytesIO()
        imsave(bytebuf, img, format='PNG')
        return fname, bytebuf.getvalue()

    bufRdd = images.rdd.map(tobuffer)
    writer = get_parallel_writer(path)(path, overwrite=overwrite, credentials=credentials())
    bufRdd.foreach(writer.write)

def totif(images, path, prefix="image", overwrite=False):
    """
    Write out TIF files for 2d or 3d image data.

    See also
    --------
    thunder.data.images.totif
    """
    dims = images.dims
    if not len(dims) in [2, 3]:
        raise ValueError("Only 2D or 3D images can be exported to tif, "
                         "images are %d-dimensional." % len(dims))

    from scipy.misc import imsave
    from io import BytesIO
    from thunder.data.writers import get_parallel_writer

    def tobuffer(kv):
        key, img = kv
        fname = prefix+"-"+"%05d.tif" % int(key)
        bytebuf = BytesIO()
        imsave(bytebuf, img, format='TIFF')
        return fname, bytebuf.getvalue()

    bufRdd = images.rdd.map(tobuffer)
    writer = get_parallel_writer(path)(path, overwrite=overwrite, credentials=credentials())
    bufRdd.foreach(writer.write)

def tobinary(images, path, prefix="image", overwrite=False):
    """
    Write out binary files for image data.

    See also
    --------
    thunder.data.images.tobinary
    """
    from thunder.data.writers import get_parallel_writer

    dims = list(asarray(images.dims.max) - asarray(images.dims.min) + 1)

    def tobuffer(kv):
        key, img = kv
        fname = prefix + "-" + "%05d.bin" % int(key)
        return fname, img.copy()

    bufRdd = images.rdd.map(tobuffer)

    writer = get_parallel_writer(path)(path, overwrite=overwrite, credentials=credentials())
    bufRdd.foreach(writer.write)
    config(path, dims=dims, dtype=images.dtype, overwrite=overwrite)

def config(path, dims, dtype='int16', name="conf.json", overwrite=True):
    """
    Helper function to write a JSON file with configuration for binary image data.
    """
    from thunder.data.writers import get_file_writer

    writer = get_file_writer(path)

    conf = {'dims': dims, 'dtype': dtype}
    confwriter = writer(path, name, overwrite=overwrite, credentials=credentials())
    confwriter.write(json.dumps(conf, indent=2))

    successwriter = writer(path, "SUCCESS", overwrite=overwrite, credentials=credentials())
    successwriter.write('')

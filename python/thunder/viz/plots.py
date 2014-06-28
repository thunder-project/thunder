from numpy import amin, amax, shape, array, maximum, float16, squeeze, reshape, transpose, zeros
from thunder.io.load import getdims, subtoind
from thunder.viz import Colorize


def spatialmap(axes, rdd, colormap='rgb', scale=1, maptype='points'):
    """Create a spatial point map or image from a collection of key-value pairs, using the
    keys as spatial indices, and the values to compute colors"""

    if maptype == 'points':
        pts = Colorize(colormap, scale).calc(rdd).collect()
        clrs = array(map(lambda (k, v): v, pts))
        x = map(lambda (k, v): k[0], pts)
        y = map(lambda (k, v): k[1], pts)
        z = map(lambda (k, v): k[2], pts)  # currently unused
        h = axes.scatter(x, y, s=100, c=clrs, alpha=0.5, edgecolor='black', linewidth=0.2)
        return axes, h

    if maptype == 'image':
        ndim = len(rdd.first()[0])
        pts = Colorize(colormap, scale).calc(rdd)
        if ndim == 3:
            proj = pts.map(lambda (k, v): ((k[0], k[1]), v)).reduceByKey(maximum).cache()
        elif ndim == 2:
            proj = pts
        else:
            raise Exception('number of spatial dimensions for images must be 2 or 3')
        dims = getdims(proj)
        proj = subtoind(proj, dims.max)
        keys = proj.map(lambda (k, _): int(k)).collect()
        im = zeros((dims.count()[0], dims.count()[1], 3))
        for i in range(0, 3):
            result = proj.map(lambda (_, v): float16(v[i])).collect()
            result = array([v for (k, v) in sorted(zip(keys, result), key=lambda (k, v): k)])
            im[:, :, i] = transpose(reshape(result, dims.count()[::-1]))
        h = axes.imshow(im)
        return axes, h

    else:
        raise Exception('mode %s not yet implemented' % maptype)


def scatter(axes, pts, colormap='rgb', scale=1):
    """Create a scatter plot of x and y points, using the values to determine
    a set of colors"""

    clrs = Colorize(colormap, scale).calc(pts)
    h = axes.scatter(pts[:, 0], pts[:, 1], s=100, c=clrs, alpha=0.5, edgecolor='black', linewidth=0.2)

    return axes, h


def tsrecon(axes, tsbase, samples):
    """Plot a single time series to represent a reconstruction using some low dimensional basis,
    and use a set of weights to create many possible lines for interactive plotting"""

    t = shape(tsbase)[1]
    tt = range(0, t)

    h = axes.plot(array(tt), 0 * array(tt), '-w', lw=5, alpha=0.5)
    axes.set_ylim(amin(tsbase)/4, amax(tsbase)/4)

    recon = map(lambda x: (x[0] * tsbase[0, :] + x[1] * tsbase[1, :]).tolist(), samples)
    topairs = lambda y: map(lambda x: list(x), zip(tt, y))
    data = map(topairs, recon)

    return axes, h, data
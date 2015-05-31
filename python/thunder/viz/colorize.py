from numpy import arctan2, sqrt, pi, abs, dstack, clip, transpose, inf, \
    random, zeros, ones, asarray, corrcoef, allclose, maximum, add, multiply, \
    nan_to_num, copy, ndarray, abs, around, ceil, sqrt


class Colorize(object):
    """
    Class for turning numerical data into colors.

    Supports a set of custom conversions (rgb, hsv, polar, and indexed)
    as well as conversions to standard matplotlib colormaps through
    either a passed colormap or a string specification.

    If vmax and vmin are not specified, numerical data will be automatically
    scaled by its maximum and minimum values.

    Supports two-dimensional and three-dimensional data.
    
    Attributes
    ----------
    cmap : string, optional, default = rainbow
        The colormap to convert to, can be one of a special set of conversions
        (rgb, hsv, hv, polar, angle, indexed), a matplotlib colormap object,
        or a string specification of a matplotlib colormap

    scale : float, optional, default = 1
        How to scale amplitude during color conversion, controls brighthness

    colors : list, optional, default = None
        List of colors for 'indexed' option

    vmin : scalar, optional, default = None
        Numerical value to set to 0 during normalization, values below will be clipped

    vmax : scalar, optional, default = None
        Numerical value to set to 1 during normalization, values above will be clipped.
    """
    def __init__(self, cmap='rainbow', scale=1, colors=None, vmin=None, vmax=None):
        self.cmap = cmap
        self.scale = scale
        self.colors = colors
        if self.cmap == 'index' and self.colors is None:
            raise Exception("Must specify colors for indexed conversion")
        self.vmin = vmin
        self.vmax = vmax

    @staticmethod
    def tile(imgs, cmap='gray', bar=False, nans=True, clim=None, grid=None, size=11):
        """
        Display a collection of images as a grid of tiles

        Parameters
        ----------
        img : list or ndarray (2D or 3D)
            The images to display. Can be a list of either 2D, 3D,
            or a mix of 2D and 3D numpy arrays. Can also be a single
            numpy array, in which case the first dimension will be assumes
            to index the image list, e.g. a (10, 512, 512) array will
            be treated as 10 different (512,512) images

        cmap : str or Colormap, optional, default = 'gray'
            A colormap to use, for non RGB images, will apply to all images

        bar : boolean, optional, default = False
            Whether to append a colorbar to each image

        nans : boolean, optional, deafult = True
            Whether to replace NaNs, if True, will replace with 0s

        clim : tuple, optional, default = None
            Limits for scaling image

        grid : tuple, optional, default = None
            Dimensions of image tile grid, if None, will use a square grid
            large enough to include all images

        size : scalar, optional, deafult = 11
            Size of the figure
        """
        from matplotlib.pyplot import figure, colorbar
        from mpl_toolkits.axes_grid1 import ImageGrid

        if not isinstance(imgs, list):
            if isinstance(imgs, ndarray):
                imgs = list(imgs)
            else:
                raise ValueError("Must provide a list of images, or an ndarray")

        imgs = [asarray(im) for im in imgs]

        if (nans is True) and (imgs[0].dtype != bool):
            imgs = [nan_to_num(im) for im in imgs]

        fig = figure(1, (size, size))

        if bar is True:
            axes_pad = 0.4
            if sum([im.ndim == 3 for im in imgs]):
                raise ValueError("Cannot show meaningful colorbar for RGB image")
            cbar_mode = "each"
        else:
            axes_pad = 0.2
            cbar_mode = None

        nimgs = len(imgs)

        if grid is None:
            c = int(ceil(sqrt(nimgs)))
            grid = (c, c)

        ngrid = grid[0] * grid[1]
        if ngrid < nimgs:
            raise ValueError("Total grid count %g too small for number of images %g" % (ngrid, nimgs))

        g = ImageGrid(fig, 111, nrows_ncols=grid, axes_pad=axes_pad,
                      cbar_mode=cbar_mode, cbar_size="5%", cbar_pad="5%")

        for i, im in enumerate(imgs):
            ax = g[i].imshow(im, cmap=cmap, interpolation='none', clim=clim)
            g[i].axis('off')

            if bar:
                cb = colorbar(ax, g[i].cax)
                rng = abs(cb.vmax - cb.vmin) * 0.05
                cb.set_ticks([around(cb.vmin + rng, 1), around(cb.vmax - rng, 1)])
                cb.outline.set_visible(False)

        if nimgs < ngrid:
            for i in range(nimgs, ngrid):
                g[i].axis('off')
                g[i].cax.axis('off')

    @staticmethod
    def image(img, cmap='gray', bar=False, nans=True, clim=None, size=9):
        """
        Streamlined display of images using matplotlib.

        Parameters
        ----------
        img : ndarray, 2D or 3D
            The image to display

        cmap : str or Colormap, optional, default = 'gray'
            A colormap to use, for non RGB images

        bar : boolean, optional, default = False
            Whether to append a colorbar

        nans : boolean, optional, deafult = True
            Whether to replace NaNs, if True, will replace with 0s

        clim : tuple, optional, default = None
            Limits for scaling image

        size : scalar, optional, deafult = 9
            Size of the figure
        """
        from matplotlib.pyplot import imshow, axis, colorbar, figure

        img = asarray(img)

        if (nans is True) and (img.dtype != bool):
            img = nan_to_num(img)

        figure(1, (size, size))

        if img.ndim == 3:
            if bar:
                raise ValueError("Cannot show meaningful colorbar for RGB images")
            if img.shape[2] != 3:
                raise ValueError("Size of third dimension must be 3 for RGB images, got %g" % img.shape[2])
            mn = img.min()
            mx = img.max()
            if mn < 0.0 or mx > 1.0:
                raise ValueError("Values must be between 0.0 and 1.0 for RGB images, got range (%g, %g)" % (mn, mx))
            imshow(img, interpolation='none', clim=clim)
        else:
            imshow(img, cmap=cmap, interpolation='none', clim=clim)

        if bar is True:
            colorbar()

        axis('off')

    def transform(self, img, mask=None, background=None, mixing=1.0):
        """
        Colorize numerical image data.

        Input can either be a single array or a list of arrays.
        Depending on the colorization option, each array must either be
        2 or 3 dimensional, see parameters for details.

        Parameters
        ----------
        img : array
            The image(s) to colorize. For rgb, hsv, polar, and indexed conversions,
            must be of shape (c, x, y, z) or (c, x, y), where c is the dimension
            containing the information for colorizing. For colormap conversions,
            must be of shape (x, y, z) or (x, y).

        mask : array
            An additional image to mask the luminance channel of the first one.
            Must be of shape (x, y, z) or (x, y), and must match dimensions of images.
            Must be strictly positive (and will be clipped below at 0).

        background : array
            An additional image to display as a grayscale background.
            Must be of shape (x, y, z) or (x, y), and must match dimensions of images.

        mixing : scalar
            If adding a background image, mixing controls the relative scale.
            Values larger than 1.0 will emphasize the background more.

        Returns
        -------
        Arrays with RGB values, with shape (x, y, z, 3) or (x, y, 3)
        """

        from matplotlib.cm import get_cmap
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap, hsv_to_rgb, Normalize

        img = asarray(img)
        dims = img.shape
        self._checkDims(dims)

        if self.cmap not in ['polar', 'angle']:

            if self.cmap in ['rgb', 'hv', 'hsv', 'indexed']:
                img = copy(img)
                for i, im in enumerate(img):
                    norm = Normalize(vmin=self.vmin, vmax=self.vmax, clip=True)
                    img[i] = norm(im)

            if isinstance(self.cmap, ListedColormap) or isinstance(self.cmap, str):
                norm = Normalize(vmin=self.vmin, vmax=self.vmax, clip=True)
                img = norm(copy(img))

        if mask is not None:
            mask = self._prepareMask(mask)
            self._checkMixedDims(mask.shape, dims)

        if background is not None:
            background = self._prepareBackground(background, mixing)
            self._checkMixedDims(background.shape, dims)

        if self.cmap == 'rgb':
            if img.ndim == 4:
                out = transpose(img, [1, 2, 3, 0])
            if img.ndim == 3:
                out = transpose(img, [1, 2, 0])

        elif self.cmap == 'hv':
            saturation = ones((dims[1], dims[2])) * 0.8
            if img.ndim == 4:
                out = zeros((dims[1], dims[2], dims[3], 3))
                for i in range(0, dims[3]):
                    out[:, :, i, :] = hsv_to_rgb(dstack((img[0][:, :, i], saturation, img[1][:, :, i])))
            if img.ndim == 3:
                out = hsv_to_rgb(dstack((img[0], saturation, img[1])))

        elif self.cmap == 'hsv':
            if img.ndim == 4:
                out = zeros((dims[1], dims[2], dims[3], 3))
                for i in range(0, dims[3]):
                    out[:, :, i, :] = hsv_to_rgb(dstack((img[0][:, :, i], img[1][:, :, i], img[2][:, :, i])))
            if img.ndim == 3:
                out = hsv_to_rgb(dstack((img[0], img[1], img[2])))

        elif self.cmap == 'angle':
            theta = ((arctan2(-img[0], -img[1]) + pi/2) % (pi*2)) / (2 * pi)
            saturation = ones((dims[1], dims[2])) * 0.8
            rho = ones((dims[1], dims[2]))
            if img.ndim == 4:
                out = zeros((dims[1], dims[2], dims[3], 3))
                for i in range(0, dims[3]):
                    out[:, :, i, :] = hsv_to_rgb(dstack((theta[:, :, i], saturation, rho)))
            if img.ndim == 3:
                out = hsv_to_rgb(dstack((theta, saturation, rho)))

        elif self.cmap == 'polar':
            theta = ((arctan2(-img[0], -img[1]) + pi/2) % (pi*2)) / (2 * pi)
            rho = sqrt(img[0]**2 + img[1]**2)
            saturation = ones((dims[1], dims[2]))
            if img.ndim == 4:
                out = zeros((dims[1], dims[2], dims[3], 3))
                for i in range(0, dims[3]):
                    out[:, :, i, :] = hsv_to_rgb(dstack((theta[:, :, i], saturation, self.scale * rho[:, :, i])))
            if img.ndim == 3:
                out = hsv_to_rgb(dstack((theta, saturation, self.scale * rho)))

        elif self.cmap == 'indexed':
            if img.ndim == 4:
                out = zeros((dims[1], dims[2], dims[3], 3))
            if img.ndim == 3:
                out = zeros((dims[1], dims[2], 3))
            for ix, clr in enumerate(self.colors):
                cmap = LinearSegmentedColormap.from_list('blend', [[0, 0, 0], clr])
                tmp = cmap(img[ix])
                if img.ndim == 4:
                    tmp = tmp[:, :, :, 0:3]
                if img.ndim == 3:
                    tmp = tmp[:, :, 0:3]
                out = maximum(out, clip(tmp, 0, 1))

        elif isinstance(self.cmap, ListedColormap):
            if img.ndim == 3:
                out = self.cmap(img)
                out = out[:, :, :, 0:3]
            if img.ndim == 2:
                out = self.cmap(img)
                out = out[:, :, 0:3]

        elif isinstance(self.cmap, str):
            func = lambda x: get_cmap(self.cmap, 256)(x)
            out = func(img)
            if img.ndim == 3:
                out = out[:, :, :, 0:3]
            if img.ndim == 2:
                out = out[:, :, 0:3]

        else:
            raise Exception('Colorization method not understood')

        out = clip(out * self.scale, 0, 1)

        if mask is not None:
            out = self.blend(out, mask, multiply)

        if background is not None:
            out = self.blend(out, background, add)

        return clip(out, 0, 1)

    @staticmethod
    def blend(img, mask, op=add):
        """
        Blend two images together using the specified operator.

        Parameters
        ----------
        img : array-like
            First image to blend

        mask : array-like
            Second image to blend

        op : func, optional, default = add
            Operator to use for combining images
        """
        if mask.ndim == 3:
            for i in range(0, 3):
                img[:, :, :, i] = op(img[:, :, :, i], mask)
        else:
            for i in range(0, 3):
                img[:, :, i] = op(img[:, :, i], mask)

        return img

    def _checkDims(self, dims):

        from matplotlib.colors import ListedColormap

        if self.cmap in ['rgb', 'hsv', 'hv', 'polar', 'angle', 'indexed']:
            if self.cmap in ['rgb', 'hsv']:
                if dims[0] != 3:
                    raise Exception('First dimension must be 3 for %s conversion' % self.cmap)
            if self.cmap in ['polar', 'angle', 'hv']:
                if dims[0] != 2:
                    raise Exception('First dimension must be 2 for %s conversion' % self.cmap)
            if self.cmap in ['indexed']:
                if dims[0] != len(self.colors):
                    raise Exception('First dimension must be %g for %s conversion with list %s'
                                    % (len(self.colors), self.cmap, self.colors))

        elif isinstance(self.cmap, ListedColormap) or isinstance(self.cmap, str):
            if len(dims) not in [2, 3]:
                raise Exception('Number of dimensions must be 2 or 3 for %s conversion' % self.cmap)

    def _checkMixedDims(self, dims1, dims2):

        from matplotlib.colors import ListedColormap

        if self.cmap in ['rgb', 'hsv', 'hv', 'polar', 'angle', 'indexed']:
            if not allclose(dims1, dims2[1:]):
                raise Exception

        elif isinstance(self.cmap, ListedColormap) or isinstance(self.cmap, str):
            if not allclose(dims1, dims2):
                raise Exception

    @staticmethod
    def _prepareMask(mask):

        mask = asarray(mask)
        mask = clip(mask, 0, inf)

        return mask / mask.max()

    @staticmethod
    def _prepareBackground(background, mixing):

        from matplotlib.colors import Normalize

        background = asarray(background)
        background = Normalize()(background)

        return background * mixing

    @classmethod
    def optimize(cls, mat, asCmap=False):
        """
        Optimal colors based on array data similarity.

        Given an (n, m) data array with n m-dimensional data points,
        tries to find an optimal set of n colors such that the similarity
        between colors in 3-dimensional space is well-matched to the similarity
        between the data points in m-dimensional space.

        Parameters
        ----------
        mat : array-like
            Array of data points to use for estimating similarity.

        asCmap : boolean, optional, default = False
            Whether to return a matplotlib colormap, if False will
            return a list of colors.
        """

        mat = asarray(mat)

        if mat.ndim < 2:
            raise Exception('Input array must be two-dimensional')

        nclrs = mat.shape[0]

        from scipy.spatial.distance import pdist, squareform
        from scipy.optimize import minimize

        distMat = squareform(pdist(mat, metric='cosine')).flatten()

        optFunc = lambda x: 1 - corrcoef(distMat, squareform(pdist(x.reshape(nclrs, 3), 'cosine')).flatten())[0, 1]
        init = random.rand(nclrs*3)
        bounds = [(0, 1) for _ in range(0, nclrs * 3)]
        res = minimize(optFunc, init, bounds=bounds, method='L-BFGS-B')
        newClrs = res.x.reshape(nclrs, 3).tolist()

        from matplotlib.colors import ListedColormap

        if asCmap:
            newClrs = ListedColormap(newClrs, name='from_list')

        return newClrs
from numpy import arctan2, sqrt, pi, array, shape, abs, dstack, clip, transpose, inf, \
    random, zeros, ones, asarray, corrcoef, allclose, amax, maximum


class Colorize(object):
    """Class for turning numerical data into colors.

    Can operate over either points or images
    
    Parameters
    ----------
    totype : string, optional, default = Pastel1
        The color to convert to

    scale : float, optional, default = 1
        How to scale amplitude during color conversion, controls brighthness

    colors : list, optional, default = None
        List of colors for 'indexed' option
    """

    def __init__(self, totype='Pastel1', scale=1, colors=None):
        self.totype = totype
        self.scale = scale
        self.colors = colors

    def points(self, pts):
        """Colorize a set of points.

        Depending on the colorization option, input must either be 1 or 2 dimensional.

        Parameters
        ----------
        pts : array
            The point or points to colorize. For rgb, polar, or hsv colorizations
            must be of shape (n, c) where c is the dimension containing
            the values for colorization and n is the number of points.
            For colormap conversions, must be of shape (n,)

        Returns
        -------
        out : array
            Color assignments for each point, shape (n, 3)
        """
        pts = asarray(pts)
        dims = pts.shape
        self._check_point_args(dims)

        from matplotlib.cm import get_cmap
        from matplotlib.colors import ListedColormap, Normalize

        if self.totype in ['rgb', 'hsv', 'polar']:
            out = map(lambda line: self.get(line), pts)
        elif isinstance(self.totype, ListedColormap) or isinstance(self.totype, str):
            if isinstance(self.totype, ListedColormap):
                out = self.totype(pts)[0:3]
            if isinstance(self.totype, str):
                norm = Normalize()
                out = get_cmap(self.totype, 256)(norm(pts))[:, 0:3]
        else:
            raise Exception('Colorization option not understood')

        return clip(out * self.scale, 0, 1)

    def get(self, pt):

        if self.totype in ['rgb', 'hsv']:
            return clip(pt, 0, inf) * self.scale

        if self.totype == 'polar':
            import colorsys
            theta = ((arctan2(-pt[0], -pt[1]) + pi/2) % (pi*2)) / (2 * pi)
            rho = sqrt(pt[0]**2 + pt[1]**2)
            return colorsys.hsv_to_rgb(theta, 1, rho * self.scale)

    def images(self, img, mask=None, background=None):
        """Colorize numerical image data.

        Input can either be a single image or a stack of images.
        Depending on the colorization option, input must either be
        2, 3, or 4 dimensional, see parameters.

        Parameters
        ----------
        img : array
            The image(s) to colorize. For rgb, hsv, and polar conversions,
            must be of shape (c, x, y, z) or (c, x, y), where
            c is the dimension containing the information for colorizing.
            For colormap conversions, must be of shape (x, y, z) or (x, y)

        mask : array
            A second image to mask the luminance channel of the first one.
            Must be of shape (x, y, z) or (x, y), and must match dimensions of images.

        background : array
            An additional image to display as a grayscale background.
            Must be of shape (x, y, z) or (x, y), and must match dimensions of images.

        Returns
        -------
        out : array
            Color assignments for images, either (x, y, z, 3) or (x, y, 3)
        """

        from matplotlib.cm import get_cmap
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap, hsv_to_rgb, Normalize

        img = asarray(img)
        img_dims = img.shape
        self._check_image_args(img_dims)

        if mask is not None:
            mask = asarray(mask)
            mask = clip(mask, 0, inf)
            mask_dims = mask.shape
            self._check_image_mask_args(mask_dims, img_dims)

        if background is not None:
            background = asarray(background)
            background = clip(background, 0, inf)
            background = 0.3 * background/amax(background)
            background_dims = background.shape
            self._check_image_mask_args(background_dims, img_dims)

        if self.totype == 'rgb':
            out = clip(img * self.scale, 0, inf)
            if img.ndim == 4:
                out = transpose(out, (1, 2, 3, 0))
            if img.ndim == 3:
                out = transpose(out, (1, 2, 0))

        elif self.totype == 'hsv':
            base = clip(img, 0, inf)
            if img.ndim == 4:
                out = zeros((img_dims[1], img_dims[2], img_dims[3], 3))
                for i in range(0, img_dims[3]):
                    out[:, :, i, :] = hsv_to_rgb(dstack((base[0][:, :, i], base[1][:, :, i], base[2][:, :, i] * self.scale)))
            if img.ndim == 3:
                out = hsv_to_rgb(dstack((base[0], base[1], base[2] * self.scale)))

        elif self.totype == 'polar':
            theta = ((arctan2(-img[0], -img[1]) + pi/2) % (pi*2)) / (2 * pi)
            rho = sqrt(img[0]**2 + img[1]**2)
            if img.ndim == 4:
                saturation = ones((img_dims[1], img_dims[2]))
                out = zeros((img_dims[1], img_dims[2], img_dims[3], 3))
                for i in range(0, img_dims[3]):
                    out[:, :, i, :] = hsv_to_rgb(dstack((theta[:, :, i], saturation, self.scale*rho[:, :, i])))
            if img.ndim == 3:
                saturation = ones((img_dims[1], img_dims[2]))
                out = hsv_to_rgb(dstack((theta, saturation, self.scale*rho)))

        elif self.totype == 'indexed':
            base = clip(img, 0, inf)
            if img.ndim == 4:
                out = zeros((img_dims[1], img_dims[2], img_dims[3], 3))
            if img.ndim == 3:
                out = zeros((img_dims[1], img_dims[2]))
            for ix, clr in enumerate(self.colors):
                cmap = LinearSegmentedColormap.from_list('blend', [[0, 0, 0], clr])
                tmp = cmap(self.scale * base[ix]/amax(base[ix]))
                if img.ndim == 4:
                    tmp = tmp[:, :, :, 0:3]
                if img.ndim == 3:
                    tmp = tmp[:, :, 0:3]
                out = maximum(out, clip(tmp, 0, 1))

        elif isinstance(self.totype, ListedColormap):
            norm = Normalize()
            func = lambda x: asarray(norm(x))
            if img.ndim == 3:
                base = func(img)
                out = self.totype(base)
                out = out[:, :, :, 0:3]
            if img.ndim == 2:
                base = func(img)
                out = self.totype(base)
                out = out[:, :, 0:3]
            out *= self.scale

        elif isinstance(self.totype, str):
            func = lambda x: get_cmap(self.totype, 256)(x)
            if img.ndim == 3:
                out = func(img)
                out = out[:, :, :, 0:3]
            if img.ndim == 2:
                out = func(img)
                out = out[:, :, 0:3]
            out *= self.scale

        else:
            raise Exception('Colorization method not understood')

        out = clip(out, 0, 1)

        if mask is not None:
            if mask.ndim == 3:
                for i in range(0, 3):
                    out[:, :, :, i] = out[:, :, :, i] * mask
            else:
                for i in range(0, 3):
                    out[:, :, i] = out[:, :, i] * mask

        if background is not None:
            if background.ndim == 3:
                for i in range(0, 3):
                    out[:, :, :, i] = out[:, :, :, i] + background
            else:
                for i in range(0, 3):
                    out[:, :, i] = out[:, :, i] + background

        return clip(out, 0, 1)

    def _check_point_args(self, dims):

        from matplotlib.colors import ListedColormap

        if self.totype in ['rgb', 'hsv', 'polar']:
            if len(dims) != 2:
                raise Exception('Number of dimensions must be 2 for %s conversion' % self.totype)
            if self.totype in ['rgb', 'hsv']:
                if dims[1] != 3:
                    raise Exception('Must have 3 values per point for %s conversion' % self.totype)
            if self.totype in ['polar']:
                if dims[1] != 2:
                    raise Exception('Must have 2 values per point for %s conversion' % self.totype)
        elif isinstance(self.totype, ListedColormap) or isinstance(self.totype, str):
            if len(dims) != 1:
                raise Exception('Number of dimensions must be 1 for %s conversion' % self.totype)

    def _check_image_args(self, dims):

        from matplotlib.colors import ListedColormap

        if self.totype in ['rgb', 'hsv', 'polar', 'indexed']:
            if len(dims) not in [3, 4]:
                raise Exception('Number of dimensions must be 3 or 4 for %s conversion' % self.totype)
            if self.totype in ['rgb', 'hsv']:
                if dims[0] != 3:
                    raise Exception('Must have 3 values per pixel for %s conversion' % self.totype)
            if self.totype in ['polar']:
                if dims[0] != 2:
                    raise Exception('Must have 2 values per pixel for %s conversion' % self.totype)
            if self.totype in ['indexed']:
                if dims[0] != len(self.colors):
                    raise Exception('Must have %g values per pixel for %s conversion with given list'
                                    % (len(self.colors), self.totype))

        elif isinstance(self.totype, ListedColormap) or isinstance(self.totype, str):
            if len(dims) not in [2, 3]:
                raise Exception('Number of dimensions must be 2 or 3 for %s conversion' % self.totype)

    def _check_image_mask_args(self, mask_dims, img_dims):

        from matplotlib.colors import ListedColormap

        if self.totype in ['rgb', 'hsv', 'polar', 'indexed']:
            if not allclose(mask_dims, img_dims[1:]):
                raise Exception

        elif isinstance(self.totype, ListedColormap) or isinstance(self.totype, str):
            if not allclose(mask_dims, img_dims):
                raise Exception

    @classmethod
    def optimize(cls, mat, ascmap=False):
        """ Optimal colors based on array data similarity.

        Given an (n, m) data array with n m-dimensional data points,
        tries to find an optimal set of n colors such that the similarity
        between colors in 3-dimensional space is well-matched to the similarity
        between the data points in m-dimensional space.

        Parameters
        ----------
        mat : array-like
            Array of data points to use for estimating similarity.

        ascmap : boolean, optional, default = False
            Whether to return a matplotlib colormap, if False will
            return a list of colors.

        """

        mat = asarray(mat)

        if mat.ndim < 2:
            raise Exception('Input array must be two-dimensional')

        nclrs = mat.shape[0]

        from scipy.spatial.distance import pdist, squareform
        from scipy.optimize import minimize

        distmat = squareform(pdist(mat, metric='cosine')).flatten()

        optfun = lambda x: 1 - corrcoef(distmat, squareform(pdist(x.reshape(nclrs, 3), 'cosine')).flatten())[0, 1]
        init = random.rand(nclrs*3)
        bounds = [(0, 1) for _ in range(0, nclrs * 3)]
        res = minimize(optfun, init, bounds=bounds)
        newclrs = res.x.reshape(nclrs, 3).tolist()

        from matplotlib.colors import ListedColormap

        if ascmap:
            newclrs = ListedColormap(newclrs, name='from_list')

        return newclrs
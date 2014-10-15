from numpy import arctan2, sqrt, pi, array, shape, abs, dstack, clip, transpose, inf, \
    random, zeros, ones, asarray, corrcoef, allclose


class Colorize(object):
    """Class for turning numerical data into colors
    Can operate over either points or images
    
    Parameters
    ----------
    totype : string, optional, default = Pastel1
        The color to convert to

    scale : float, optional, default = 1
        How to scale amplitude during color conversion, controls brighthness
    """

    def __init__(self, totype='Pastel1', scale=1):
        self.totype = totype
        self.scale = scale

    def points(self, pts):
        """Colorize a set of points or a single points. Input must have
        either one dimension (a single point) or two dimensions (a list or
        array of points). 

        Parameters
        ----------
        pts : array
            The point or points to colorize. Must be of shape (n, c) or (c,) where
            c is the dimension containing the information for colorizing. 

        Returns
        -------
        out : array
            Color assignments for each point, either (n, 3) or (3,)
        """
        pts = array(pts)
        n = len(pts[0])
        self.checkargs(n)

        if pts.ndim > 2:
            raise Exception("points must have 1 or 2 dimensions")

        if pts.ndim == 1:
            out = clip(self.get(pts), 0, 1)
        else:
            out = map(lambda line: clip(self.get(line), 0, 1), pts)

        return out

    def get(self, line):

        from matplotlib.cm import get_cmap

        if (self.totype == 'rgb') or (self.totype == 'hsv'):
            return abs(line) * self.scale

        if self.totype == 'polar':
            import colorsys
            theta = ((arctan2(-line[0], -line[1]) + pi/2) % (pi*2)) / (2 * pi)
            rho = sqrt(line[0]**2 + line[1]**2)
            return colorsys.hsv_to_rgb(theta, 1, rho * self.scale)

        else:
            return get_cmap(self.totype, 256)(line[0] * self.scale)[0:3]

    def images(self, img, mask=None):
        """Colorize numerical image data.

        Input can either be a single image or a stack of images.
        Depending on the colorization option, input must either be

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

        Returns
        -------
        out : array
            Color assignments for images, either (x, y, z, 3) or (x, y, 3)
        """

        from matplotlib.cm import get_cmap
        from matplotlib.colors import ListedColormap, hsv_to_rgb, Normalize

        img = asarray(img)
        img_dims = img.shape
        self._check_image_args(img_dims)

        if mask is not None:
            mask = asarray(mask)
            mask = clip(mask, 0, inf)
            mask_dims = mask.shape
            self._check_image_mask_args(mask_dims, img_dims)

        if (self.totype == 'rgb') or (self.totype == 'hsv'):
            out = clip(img, 0, inf)
            if img.ndim == 4:
                out = transpose(out, (1, 2, 3, 0))
            if img.ndim == 3:
                out = transpose(out, (1, 2, 0))

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

        elif isinstance(self.totype, str):
            func = lambda x: get_cmap(self.totype, 256)(x)
            if img.ndim == 3:
                out = func(img)
                out = out[:, :, :, 0:3]
            if img.ndim == 2:
                out = func(img)
                out = out[:, :, 0:3]
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

        return clip(out * self.scale, 0, 1)

    def _check_image_args(self, dims):

        from matplotlib.colors import ListedColormap

        if self.totype in ['rgb', 'hsv', 'polar']:
            if len(dims) not in [3, 4]:
                raise Exception('Number of dimensions must be 3 or 4 for %s conversion' % self.totype)
            if self.totype in ['rgb', 'hsv']:
                if dims[0] != 3:
                    raise Exception('Must have 3 values per pixel for %s conversion' % self.totype)
            if self.totype in ['polar']:
                if dims[0] != 2:
                    raise Exception('Must have 2 values per pixel for %s conversion' % self.totype)

        elif isinstance(self.totype, ListedColormap) or isinstance(self.totype, str):
            if len(dims) not in [2, 3]:
                raise Exception('Number of dimensions must be 2 or 3 for %s conversion' % self.totype)

    def _check_image_mask_args(self, mask_dims, img_dims):

        from matplotlib.colors import ListedColormap

        if self.totype in ['rgb', 'hsv', 'polar']:
            if not allclose(mask_dims[1:], img_dims):
                raise Exception

        if isinstance(self.totype, ListedColormap) or isinstance(self.totype, str):
            if not allclose(mask_dims, img_dims):
                raise Exception

    @classmethod
    def optimize(cls, mat, ascmap=False):

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
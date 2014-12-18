"""
Classes for mass-unvariate regression
"""

from numpy import sum, outer, inner, mean, shape, dot, transpose, concatenate, ones, asarray

from thunder.rdds.series import Series
from thunder.utils.common import loadMatVar, pinv


class RegressionModel(object):
    """
    Base class for loading and fitting regression models.

    See also
    --------
    MeanRegressionModel : simple averaging via regression
    LinearRegressionModel : linear regression
    BilinearRegressionModel : bilinear regression
    """

    @staticmethod
    def load(modelFile, regressMode, **opts):
        from thunder.utils.common import checkParams
        checkParams(regressMode.lower(), REGRESSION_MODELS.keys())
        return REGRESSION_MODELS[regressMode.lower()](modelFile, **opts)

    def get(self, y):
        pass

    def fit(self, data, comps=None):
        """
        Fit mass univariate regression models

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            The data to fit regression models to, a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        Returns
        -------
        result : Series
            Fitted model parameters: betas, summary statistic, and residuals
        """

        if not (isinstance(data, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if comps is not None:
            traj = data.rdd.map(lambda (_, v): v).map(
                lambda x: outer(x, inner(self.get(x)[0] - mean(self.get(x)[0]), comps))).sum() / data.count()
            return traj
        else:
            result = Series(data.rdd.mapValues(lambda x: self.get(x)),
                            index=['betas', 'stats', 'resid']).__finalize__(data)
            return result


class MeanRegressionModel(RegressionModel):
    """
    Regression in the form of simple averaging.

    Multiplies data by a binary design matrix.

    Parameters
    ----------
    modelFile : array, or string
        Array contaiing design matrix, or location of a MAT file

    var : string, default = 'X'
        Variable name if loading from a MAT file

    Attributes
    ----------
    x : array
        The design matrix

    xHat : array
        Pseudoinverse of the design matrix
    """

    def __init__(self, modelFile, var='X'):
        if type(modelFile) is str:
            x = loadMatVar(modelFile, var)
        else:
            x = modelFile
        x = x.astype(float)
        xHat = (x.T / sum(x, axis=1)).T
        self.x = x
        self.xHat = xHat

    def get(self, y):
        """Compute regression coefficients, r2 statistic, and residuals"""

        b = dot(self.xHat, y)
        predic = dot(b, self.x)
        resid = y - predic
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        if sst == 0:
            r2 = 0
        else:
            r2 = 1 - sse / sst
        return asarray([b, r2, resid])


class LinearRegressionModel(RegressionModel):
    """
    Ordinary least squares linear regression.

    Parameters
    ----------
    modelFile : array, or string
        Array contaiing design matrix, or location of a MAT file

    var : string, default = 'X'
        Variable name if loading from a MAT file

    Attributes
    ----------
    x : array
        The design matrix

    xhat : array
        Pseudoinverse of the design matrix
    """

    def __init__(self, modelFile, var='X'):
        if isinstance(modelFile, basestring):
            x = loadMatVar(modelFile, var)
        else:
            x = modelFile
        x = concatenate((ones((1, shape(x)[1])), x))
        xHat = pinv(x)
        self.x = x
        self.xHat = xHat

    def get(self, y):
        """Compute regression coefficients, r2 statistic, and residuals"""

        b = dot(self.xHat, y)
        predic = dot(b, self.x)
        resid = y - predic
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        if sst == 0:
            r2 = 0
        else:
            r2 = 1 - sse / sst
        return asarray([b[1:], r2, resid])


class BilinearRegressionModel(RegressionModel):
    """
    Bilinear regression with two design matrices.

    Parameters
    ----------
    modelFile : tuple(array), or tuple(string)
        Tuple of arrays contaiing design matrices,
        or locations of MAT files

    var : list(string), default = ('X1','X2')
        Variable names if loading from MAT files

    Attributes
    ----------
    x1 : array
        The first design matrix

    x2 : array
        The second design matrix

    x1hat : array
        Pseudoinverse of the first design matrix
    """

    def __init__(self, modelFile, var=('X1', 'X2')):
        if isinstance(modelFile, basestring):
            x1 = loadMatVar(modelFile[0], var[0])
            x2 = loadMatVar(modelFile[1], var[1])
        else:
            x1 = modelFile[0]
            x2 = modelFile[1]
        x1Hat = pinv(x1)
        self.x1 = x1
        self.x2 = x2
        self.x1Hat = x1Hat

    def get(self, y):
        """Compute regression coefficients from the second design matrix,
        a single r2 statistic, and residuals for the full model"""

        b1 = dot(self.x1Hat, y)
        b1 = b1 - min(b1)
        b1Hat = dot(transpose(self.x1), b1)
        if sum(b1Hat) == 0:
            b1Hat += 1E-06
        x3 = self.x2 * b1Hat
        x3 = concatenate((ones((1, shape(x3)[1])), x3))
        x3Hat = pinv(x3)
        b2 = dot(x3Hat, y)
        predic = dot(b2, x3)
        resid = y - predic
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        if sst == 0:
            r2 = 0
        else:
            r2 = 1 - sse / sst

        return asarray([b2[1:], r2, resid])

REGRESSION_MODELS = {
    'linear': LinearRegressionModel,
    'bilinear': BilinearRegressionModel,
    'mean': MeanRegressionModel
}

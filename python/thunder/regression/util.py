"""
utilities for regression and model fitting
"""

from scipy.io import loadmat
from numpy import array, sum, outer, inner, mean, shape, dot, transpose, concatenate, ones, angle, abs, exp
from scipy.linalg import inv


class RegressionModel(object):
    """class for loading and fitting a regression"""

    @staticmethod
    def load(modelfile, regressmode, *opts):
        return REGRESSION_MODELS[regressmode](modelfile, *opts)

    def get(self, y):
        pass

    def fit(self, data, comps=None):
        if comps is not None:
            traj = data.map(
                lambda x: outer(x, inner(self.get(x)[0] - mean(self.get(x)[0]), comps))).reduce(
                    lambda x, y: x + y) / data.count()
            return traj
        else:
            result = data.map(lambda x: self.get(x))
            betas = result.map(lambda x: x[0])
            stats = result.map(lambda x: x[1])
            resid = result.map(lambda x: x[2])
            return betas, stats, resid


class LinearRegressionModel(RegressionModel):
    """class for linear regression"""

    def __init__(self, modelfile):
        """load model. modelfile can be an array, or a string
        if its a string, assumes design matrix is a MAT file
        with name modelfile_X
        """
        if type(modelfile) is str:
            x = loadmat(modelfile + "_X.mat")['X']
        else:
            x = modelfile
        x = concatenate((ones((1, shape(x)[1])), x))
        x_hat = dot(inv(dot(x, transpose(x))), x)
        self.x = x
        self.x_hat = x_hat

    def get(self, y):
        """compute regression coefficients, r2 statistic, and residuals"""

        b = dot(self.x_hat, y)
        predic = dot(b, self.x)
        resid = y - predic
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        if sst == 0:
            r2 = 0
        else:
            r2 = 1 - sse / sst
        return b[1:], r2, resid


class BilinearRegressionModel(RegressionModel):
    """class for bilinear regression"""

    def __init__(self, modelfile):
        """load model. modelfile can be a tuple of arrays, or a string
        if its a string, assumes two design matrices are MAT files
        with names modelfile_X1 and modefile_X2
        """
        if type(modelfile) is str:
            x1 = loadmat(modelfile + "_X1.mat")['X1']
            x2 = loadmat(modelfile + "_X2.mat")['X2']
        else:
            x1 = modelfile[0]
            x2 = modelfile[1]
        x1_hat = dot(inv(dot(x1, transpose(x1))), x1)
        self.x1 = x1
        self.x2 = x2
        self.x1_hat = x1_hat

    def get(self, y):
        """compute two sets of regression coefficients, r2 statistic, and residuals"""

        b1 = dot(self.x1_hat, y)
        b1 = b1 - min(b1)
        b1_hat = dot(transpose(self.x1), b1)
        if sum(b1_hat) == 0:
            b1_hat += 1E-06
        x3 = self.x2 * b1_hat
        x3 = concatenate((ones((1, shape(x3)[1])), x3))
        x3_hat = dot(inv(dot(x3, transpose(x3))), x3)
        b2 = dot(x3_hat, y)
        predic = dot(b2, x3)
        resid = y - predic
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        if sst == 0:
            r2 = 0
        else:
            r2 = 1 - sse / sst

        return b2[1:], r2, resid


class TuningModel(object):
    """class for loading and fitting a tuning model"""

    def __init__(self, modelfile):
        """load model. modelfile can be an array, or a string
        if it's a string, assumes stim is a MAT file
        at with name modelfile_s
        """
        if type(modelfile) is str:
            self.s = loadmat(modelfile + "_s.mat")['s']
        else:
            self.s = modelfile

    @staticmethod
    def load(modelfile, tuningmode):
        return TUNING_MODELS[tuningmode](modelfile)

    def get(self, y):
        pass

    def fit(self, data):
        return data.map(lambda x: self.get(x))


class CircularTuningModel(TuningModel):
    """class for circular tuning"""

    def get(self, y):
        """estimates the circular mean and variance ("kappa")
        identical to the max likelihood estimates of the
        parameters of the best fitting von-mises function
        """
        y = y - min(y)
        if sum(y) == 0:
            y += 1E-06
        y = y / sum(y)
        r = inner(y, exp(1j * self.s))
        mu = angle(r)
        v = abs(r) / sum(y)
        if v < 0.53:
            k = 2 * v + (v ** 3) + 5 * (v ** 5) / 6
        elif (v >= 0.53) & (v < 0.85):
            k = -.4 + 1.39 * v + 0.43 / (1 - v)
        elif (v ** 3 - 4 * (v ** 2) + 3 * v) == 0:
            k = array([0.0])
        else:
            k = 1 / (v ** 3 - 4 * (v ** 2) + 3 * v)
        if k > 1E8:
            k = array([0.0])
        return mu, k


class GaussianTuningModel(TuningModel):
    """class for gaussian tuning"""

    def get(self, y):
        """estimates the mean and variance
        similar to the max likelihood estimates of the
        parameters of the best fitting gaussian
        but non-infinite supports may bias estimates
        """
        y[y < 0] = 0
        if sum(y) == 0:
            y += 1E-06
        y = y / sum(y)
        mu = dot(self.s, y)
        sigma = dot((self.s - mu) ** 2, y)
        return mu, sigma


TUNING_MODELS = {
    'circular': CircularTuningModel,
    'gaussian': GaussianTuningModel
}

REGRESSION_MODELS = {
    'linear': LinearRegressionModel,
    'bilinear': BilinearRegressionModel
}
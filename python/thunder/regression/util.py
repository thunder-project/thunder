"""
Utilities for regression and model fitting
"""

from scipy.io import *
from numpy import *
from scipy.linalg import *


class RegressionModel(object):
    @staticmethod
    def load(modelFile, regressMode, *opts):
        return REGRESSION_MODELS[regressMode](modelFile, *opts)

    def get(self, y):
        pass

    def fit(self, data, comps=None):
        if comps is not None:
            traj = data.map(
                lambda x: outer(x, inner(self.get(x)[0] - mean(self.get(x)[0]), comps))).reduce(
                lambda x, y: x + y) / data.count()
            return traj
        else:
            betas = data.map(lambda x: self.get(x))
            return betas


class CrossCorrModel(RegressionModel):
    def __init__(self, modelFile, mxLag):
        X = loadmat(modelFile + "_X.mat")['X']
        X = X - mean(X)
        X = X / norm(X)
        if mxLag is not 0:
            shifts = range(-mxLag, mxLag+1)
            d = shape(X)[1]
            m = len(shifts)
            shiftedX = zeros((m, d))
            for ix in range(0, len(shifts)):
                shiftedX[ix, :] = roll(X, ix)
            self.X = shiftedX
        else:
            self.X = X
        print(self.X)

    def get(self, y):
        y = y - mean(y)
        y /= norm(y) + 0.0001
        b = dot(self.X, y)
        return b


class MeanRegressionModel(RegressionModel):
    def __init__(self, modelFile):
        self.X = loadmat(modelFile + "_X.mat")['X'].astype(float)

    def get(self, y):
        b = dot(self.X, y)
        return b, 1


class LinearRegressionModel(RegressionModel):
    def __init__(self, modelFile):
        X = loadmat(modelFile + "_X.mat")['X']
        X = concatenate((ones((1, shape(X)[1])), X))
        X = X.astype(float)
        g = loadmat(modelFile + "_g.mat")['g']
        g = g.astype(float)[0]
        Xhat = dot(inv(dot(X, transpose(X))), X)
        self.X = X
        self.Xhat = Xhat
        self.g = g
        self.nG = len(unique(self.g))

    def get(self, y):
        b = dot(self.Xhat, y)
        predic = dot(b, self.X)
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        r2 = 1 - sse / sst
        return (b[1:], r2)


class LinearShuffleRegressionModel(LinearRegressionModel):
    def __init__(self, modelFile):
        super(LinearShuffleRegressionModel, self).__init__(modelFile)
        self.nRnd = float(2)

    def get(self, y):
        b = dot(self.Xhat, y)
        predic = dot(b, self.X)
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        r2 = 1 - sse / sst
        r2shuffle = zeros((self.nRnd,))
        X = copy(self.X)
        m = shape(X)[1]
        for iShuf in range(0, int(self.nRnd)):
            for ix in range(0, shape(X)[0]):
                shift = int(round(random.rand(1) * m))
                X[ix, :] = roll(X[ix, :], shift)
            b = lstsq(transpose(X), y)[0]
            predic = dot(b, X)
            sse = sum((predic - y) ** 2)
            r2shuffle[iShuf] = 1 - sse / sst
        p = sum(r2shuffle > r2) / self.nRnd
        return (b[1:], [r2, p])


class BilinearRegressionModel(RegressionModel):
    def __init__(self, modelFile):
        X1 = loadmat(modelFile + "_X1.mat")['X1']
        X2 = loadmat(modelFile + "_X2.mat")['X2']
        X1hat = dot(inv(dot(X1, transpose(X1))), X1)
        self.X1 = X1
        self.X2 = X2
        self.X1hat = X1hat

    def get(self, y):
        b1 = dot(self.X1hat, y)
        b1 = b1 - min(b1)
        b1hat = dot(transpose(self.X1), b1)
        if sum(b1hat) == 0:
            b1hat = b1hat + 0.001
        X3 = self.X2 * b1hat
        X3 = concatenate((ones((1, shape(X3)[1])), X3))
        X3hat = dot(inv(dot(X3, transpose(X3))), X3)
        b2 = dot(X3hat, y)
        predic = dot(b2, X3)
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        r2 = 1 - sse / sst

        return (b2[1:], r2, b1)


class ShotgunRegressionModel(RegressionModel):
    def __init__(self, modelFile):
        y = loadmat(modelFile + "_y.mat")['y']
        y = y.astype(float)
        #y = (y - mean(y)) / std(y)
        if shape(y)[0] == 1:
            y = transpose(y)
        self.y = y


class TuningModel(object):
    def __init__(self, modelFile):
        self.s = loadmat(modelFile + "_s.mat")['s']

    @staticmethod
    def load(modelFile, tuningMode):
        return TUNING_MODELS[tuningMode](modelFile)

    def get(self, y):
        pass

    def fit(self, data):
        return data.map(lambda x: self.get(x))

    def curves(self, data):
        def inRange(val, rng1, rng2):
            if (val > rng1) & (val < rng2):
                return True
            else:
                return False

        vals = linspace(amin(self.s), amax(self.s), 4)
        means = zeros((len(vals) - 1, max(shape(self.s))))
        sds = zeros((len(vals) - 1, max(shape(self.s))))
        for iv in range(0, len(vals) - 1):
            subset = data.filter(
                lambda b: (b[1] > 0.005) & inRange(self.get(b[0])[0], vals[iv], vals[iv + 1]))
            n = subset.count()
            means[iv, :] = subset.map(lambda b: b[0]).reduce(lambda x, y: x + y) / n
            sds[iv, :] = subset.map(lambda b: (b[0] - means[iv, :]) ** 2).reduce(
                lambda x, y: x + y) / (n - 1)

        return means, sds


class CircularTuningModel(TuningModel):
    def get(self, y):
        y = y - min(y)
        y = y / sum(y)
        r = inner(y, exp(1j * self.s))
        mu = angle(r)
        v = absolute(r) / sum(y)
        if v < 0.53:
            k = 2 * v + (v ** 3) + 5 * (v ** 5) / 6
        elif (v >= 0.53) & (v < 0.85):
            k = -.4 + 1.39 * v + 0.43 / (1 - v)
        else:
            k = 1 / (v ** 3 - 4 * (v ** 2) + 3 * v)
        return (mu, k)


class GaussianTuningModel(TuningModel):
    def get(self, y):
        y[y < 0] = 0
        y = y / sum(y)
        mu = dot(self.s, y)
        sigma = dot((self.s - mu) ** 2, y)
        return (mu, sigma)


TUNING_MODELS = {
    'circular': CircularTuningModel,
    'gaussian': GaussianTuningModel
}

REGRESSION_MODELS = {
    'mean': MeanRegressionModel,
    'linear': LinearRegressionModel,
    'linear-shuffle': LinearShuffleRegressionModel,
    'crosscorr': CrossCorrModel,
    'bilinear': BilinearRegressionModel,
    'shotgun': ShotgunRegressionModel,
}
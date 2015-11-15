import pytest
from numpy import array, allclose, r_, pi, mean, std, zeros, sqrt
from sklearn import linear_model as lm

from thunder.regression import LinearRegression
from thunder.regression import TuningModel
from thunder.data.series.readers import fromList

pytestmark = pytest.mark.usefixtures("context")

X = array([
    [-0.4309741,   0.43440693,  0.19946369,  1.40428728],
    [0.54587086, -1.1092286,  -0.27258427,  0.35205421],
    [-0.4432777,   0.40580108,  0.20938645,  0.26480389],
    [-0.53239659, -0.90966912, -0.13967252,  1.38274305],
    [0.35731376,  0.39878607,  0.07762888,  1.82299252],
    [0.36687294, -0.17079843, -0.17765573,  0.87161138],
    [0.3017848,   1.36537541,  0.91211512, -0.80570055],
    [-0.72330999,  0.36319617,  0.08986615, -0.7830115],
    [1.11477831,  0.41631623,  0.11104172, -0.90049209],
    [-1.62162968,  0.46928843,  0.62996118,  1.08668594]
])

y0 = array([
    4.57058016, -4.06400691,  4.25957933,  2.01583617,  0.34791879,
    -0.9113852, 3.41167194,  5.26059279, -2.35116878,  6.28263909
])

tol = 1E-3


def test_ordinary_linear_regression():
    y = fromList([y0])

    # sklearn's normalize=True option 'unscales' after fitting, we do not, so we do the scaling for them
    Xscaled = (X - mean(X, axis=0))/std(X, axis=0, ddof=1)

    # no intercept, no scaling
    betas = LinearRegression(intercept=False).fit(X, y).coeffs.values().first()
    betas0 = lm.LinearRegression(fit_intercept=False).fit(X, y0).coef_
    assert allclose(betas, betas0, atol=tol)

    # intercept, no scaling
    betas = LinearRegression().fit(X, y).coeffs.values().first()
    result = lm.LinearRegression(fit_intercept=True).fit(X, y0)
    betas0 = r_[result.intercept_, result.coef_]
    assert allclose(betas, betas0, atol=tol)

    # no intercept, scaling
    betas = LinearRegression(intercept=False, zscore=True).fit(X, y).coeffs.values().first()
    betas0 = lm.LinearRegression(fit_intercept=False).fit(Xscaled, y0).coef_
    assert allclose(betas, betas0, atol=tol)

    # intercept, scaling
    betas = LinearRegression(zscore=True).fit(X, y).coeffs.values().first()
    result = lm.LinearRegression().fit(Xscaled, y0)
    betas0 = r_[result.intercept_, result.coef_]
    assert allclose(betas, betas0, atol=tol)


def test_tikhonov_linear_regression():
    y = fromList([y0])
    R = array([[1, -2, 1, 0], [0, 1, -2, 0]], dtype='float')
    c = 2.0
    # no intercept, no normalization
    # tikhonov regularization can be recast in the form of OLS regression with an augmented design matrix
    Xaug = r_[X, sqrt(c)*R]
    yaug = r_[y0, zeros(R.shape[0])]

    betas = LinearRegression('tikhonov', intercept=False,  R=R, c=c).fit(X, y).coeffs.values().first()
    betas0 = lm.LinearRegression(fit_intercept=False).fit(Xaug, yaug).coef_
    assert allclose(betas, betas0, atol=tol)

    # intercept, no normalization
    Xscaled = (X - mean(X, axis=0))/std(X, axis=0, ddof=1)
    Xaug = r_[Xscaled, sqrt(c)*R]
    yint = mean(y0)
    yaug = r_[y0-yint, zeros(R.shape[0])]

    betas = LinearRegression('tikhonov', zscore=True, R=R, c=c).fit(X, y).coeffs.values().first()
    result = lm.LinearRegression(fit_intercept=False).fit(Xaug, yaug)
    betas0 = r_[yint, result.coef_]
    assert allclose(betas, betas0, atol=tol)


def test_ridge_linear_regression():
    y = fromList([y0])
    c = 2.0
    betas = LinearRegression('ridge', intercept=False, c=c).fit(X, y).coeffs.values().first()
    betas0 = lm.Ridge(fit_intercept=False, alpha=c).fit(X, y0).coef_
    assert(allclose(betas, betas0, atol=tol))


def test_linear_regression_model_methods():
    y = fromList([y0])
    model = LinearRegression().fit(X, y)
    model0 = lm.LinearRegression().fit(X, y0)

    yhat = model.predict(X).values().first()
    yhat0 = model0.predict(X)
    assert allclose(yhat, yhat0, atol=tol)

    R2_initial = model.stats.values().first()
    R2_score = model.score(X, y).values().first()
    R2_0 = model0.score(X, y0)
    assert allclose(R2_initial, R2_score, atol=tol)
    assert allclose(R2_initial, R2_0, atol=tol)

    result1, result2 = model.predictAndScore(X, y)
    yhat = result1.values().first()
    R2 = result2.values().first()
    assert allclose(yhat, yhat0, atol=tol)
    assert allclose(R2, R2_0, atol=tol)


def test_gaussian_tuning_model():
    data = fromList([array([1.5, 2.3, 6.2, 5.1, 3.4, 2.1])])
    s = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    model = TuningModel.load(s, "gaussian")
    params = model.fit(data)
    assert allclose(params.select('center').values().collect()[0], array([0.36262]), atol=tol)
    assert allclose(params.select('spread').values().collect()[0], array([0.01836]), atol=tol)


def test_circular_tuning_model():
    data = fromList([array([1.5, 2.3, 6.2, 5.1, 3.4, 2.1])])
    s = array([-pi/2, -pi/3, -pi/4, pi/4, pi/3, pi/2])
    model = TuningModel.load(s, "circular")
    params = model.fit(data)
    assert allclose(params.select('center').values().collect()[0], array([0.10692]), atol=tol)
    assert allclose(params.select('spread').values().collect()[0], array([1.61944]), atol=tol)

import pytest
from numpy import array, allclose, transpose, random, dot, corrcoef, diag
from numpy.linalg import norm, inv

from thunder.factorization.ica import ICA
from thunder.factorization.svd import SVD
from thunder.factorization.nmf import NMF
from thunder.factorization.pca import PCA
from thunder.data.series.readers import fromList

pytestmark = pytest.mark.usefixtures("context")


def test_pca():
    dataLocal = [
        array([1.0, 1.0, 1.0, 5.0]),
        array([2.0, 3.0, 4.0, 1.0]),
        array([6.0, 0.0, 6.0, 6.0])
    ]
    mat = fromList(dataLocal)

    pca1 = PCA(k=1, svdMethod='direct')
    pca1.fit(mat)
    out1_comps = pca1.comps
    out1_scores = pca1.scores.collectValuesAsArray() * pca1.latent
    out1_transform_scores = pca1.transform(mat).collectValuesAsArray() * pca1.latent

    from sklearn.decomposition import PCA as skPCA
    pca2 = skPCA(n_components=1)
    pca2.fit(array(dataLocal))
    out2_comps = pca2.components_
    out2_scores = pca2.transform(array(dataLocal))

    assert allclose(out1_comps, out2_comps) | allclose(out1_comps, -out2_comps)
    assert allclose(out1_scores, out2_scores) | allclose(out1_scores, -out2_scores)
    assert allclose(out1_scores, out1_transform_scores)


def test_svd_direct():
    dataLocal = [
        array([1.0, 2.0, 6.0]),
        array([1.0, 3.0, 0.0]),
        array([1.0, 4.0, 6.0]),
        array([5.0, 1.0, 4.0])
    ]
    mat = fromList(dataLocal)

    svd = SVD(k=1, method="direct")
    svd.calc(mat)
    uTest = transpose(array(svd.u.rows().collect()))[0]
    vTest = svd.v[0]

    from scipy.linalg import svd as ssvd
    uTrue, sTrue, vTrue = ssvd(array(dataLocal))

    assert allclose(svd.s[0], sTrue[0])
    assert allclose(vTest, vTrue[0, :]) | allclose(-vTest, vTrue[0, :])
    assert allclose(uTest, uTrue[:, 0]) | allclose(-uTest, uTrue[:, 0])


def test_svd_em():
    dataLocal = [
        array([1.0, 2.0, 6.0]),
        array([1.0, 3.0, 0.0]),
        array([1.0, 4.0, 6.0]),
        array([5.0, 1.0, 4.0])
    ]
    mat = fromList(dataLocal)

    svd = SVD(k=1, method="em")
    svd.calc(mat)
    uTest = transpose(array(svd.u.rows().collect()))[0]
    vTest = svd.v[0]

    from scipy.linalg import svd as ssvd
    uTrue, sTrue, vTrue = ssvd(array(dataLocal))

    tol = 10e-04  # allow small error for iterative method
    assert allclose(svd.s[0], sTrue[0], atol=tol)
    assert allclose(vTest, vTrue[0, :], atol=tol) | allclose(-vTest, vTrue[0, :], atol=tol)
    assert allclose(uTest, uTrue[:, 0], atol=tol) | allclose(-uTest, uTrue[:, 0], atol=tol)


def test_svd_conversion():
    dataLocal = [
        array([1.0, 2.0, 6.0]),
        array([1.0, 3.0, 0.0]),
        array([1.0, 4.0, 6.0]),
        array([5.0, 1.0, 4.0])
    ]
    mat = fromList(dataLocal)
    SVD(k=1, method='direct').calc(mat)


def test_ica():
    random.seed(42)
    data, s, a = ICA.make(shape=(100, 10), withparams=True)

    ica = ICA(c=2, svdMethod="direct", seed=1)
    ica.fit(data)

    s_ = array(ica.sigs.rows().collect())

    # test accurate recovery of original signals
    tol = 0.01
    assert allclose(abs(corrcoef(s[:, 0], s_[:, 0])[0, 1]), 1, atol=tol) or \
        allclose(abs(corrcoef(s[:, 0], s_[:, 1])[0, 1]), 1, atol=tol)
    assert allclose(abs(corrcoef(s[:, 1], s_[:, 0])[0, 1]), 1, atol=tol) or \
        allclose(abs(corrcoef(s[:, 1], s_[:, 1])[0, 1]), 1, atol=tol)

    # test accurate reconstruction from sources
    assert allclose(array(data.values().collect()), dot(s_, ica.a.T))


def test_nmf_als():
    #  set data and initializing constants
    dataLocal = array([
        [1.0, 2.0, 6.0],
        [1.0, 3.0, 0.0],
        [1.0, 4.0, 6.0],
        [5.0, 1.0, 4.0]])
    mat = fromList(dataLocal)
    h0 = array(
        [[0.09082617,  0.85490047,  0.57234593],
         [0.82766740,  0.21301186,  0.90913979]])

    # if the rows of h are not normalized on each iteration:
    hTrue = array(
        [[0, 0.6010, 0.9163],
         [0.8970, 0.1556, 0.7423]])
    wTrue = array(
        [[4.5885, 1.5348],
         [1.3651, 0.2184],
         [5.9349, 1.0030],
         [0, 5.5147]])

    # if the columns of h are normalized (as in the current implementation):
    scaleMat = diag(norm(hTrue, axis=1))
    hTrue = dot(inv(scaleMat), hTrue)
    wTrue = dot(wTrue, scaleMat)

    # calculate NMF using the Thunder implementation
    # (maxiter=9 corresponds with Matlab algorithm)
    nmfThunder = NMF(k=2, method="als", h0=h0, maxIter=9)
    nmfThunder.fit(mat)
    hThunder = nmfThunder.h
    wThunder = array(nmfThunder.w.values().collect())

    tol = 1e-03  # allow small error
    assert allclose(wThunder, wTrue, atol=tol)
    assert allclose(hThunder, hTrue, atol=tol)


def test_nmf_init():
    dataLocal = array([
        [1.0, 2.0, 6.0],
        [1.0, 3.0, 0.0],
        [1.0, 4.0, 6.0],
        [5.0, 1.0, 4.0]])
    mat = fromList(dataLocal)

    nmfThunder = NMF(k=2, reconHist='final')
    nmfThunder.fit(mat)
    assert nmfThunder.reconErr < 2.9950
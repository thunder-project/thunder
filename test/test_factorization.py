import pytest
from numpy import array, allclose, transpose, random, dot, corrcoef, diag
from numpy.linalg import norm, inv

from thunder.factorization.ica import ICA
from thunder.factorization.svd import SVD
from thunder.factorization.nmf import NMF
from thunder.factorization.pca import PCA
from thunder.data.series.readers import fromlist

pytestmark = pytest.mark.usefixtures("context")


def test_pca():
    local = [
        array([1.0, 1.0, 1.0, 5.0]),
        array([2.0, 3.0, 4.0, 1.0]),
        array([6.0, 0.0, 6.0, 6.0])
    ]
    mat = fromlist(local)
    pca1 = PCA(k=1, svdmethod='direct')
    pca1.fit(mat)
    out1_comps = pca1.comps
    out1_scores = pca1.scores.toarray() * pca1.latent
    out1_transform_scores = pca1.transform(mat).toarray() * pca1.latent
    from sklearn.decomposition import PCA as skPCA
    pca2 = skPCA(n_components=1)
    pca2.fit(array(local))
    out2_comps = pca2.components_
    out2_scores = pca2.transform(array(local)).squeeze()
    print(out1_scores)
    print(out2_scores)
    assert allclose(out1_comps, out2_comps) | allclose(out1_comps, -out2_comps)
    assert allclose(out1_scores, out2_scores) | allclose(out1_scores, -out2_scores)
    assert allclose(out1_scores, out1_transform_scores)


def test_svd_direct():
    local = [
        array([1.0, 2.0, 6.0]),
        array([1.0, 3.0, 0.0]),
        array([1.0, 4.0, 6.0]),
        array([5.0, 1.0, 4.0])
    ]
    mat = fromlist(local)
    svd = SVD(k=1, method="direct")
    svd.calc(mat)
    utest = transpose(array(svd.u.rows().collect()))[0]
    vtest = svd.v[0]
    from scipy.linalg import svd as ssvd
    utrue, strue, vtrue = ssvd(array(local))
    assert allclose(svd.s[0], strue[0])
    assert allclose(vtest, vtrue[0, :]) | allclose(-vtest, vtrue[0, :])
    assert allclose(utest, utrue[:, 0]) | allclose(-utest, utrue[:, 0])


def test_svd_em():
    local = [
        array([1.0, 2.0, 6.0]),
        array([1.0, 3.0, 0.0]),
        array([1.0, 4.0, 6.0]),
        array([5.0, 1.0, 4.0])
    ]
    mat = fromlist(local)
    svd = SVD(k=1, method="em")
    svd.calc(mat)
    utest = transpose(array(svd.u.rows().collect()))[0]
    vtest = svd.v[0]
    from scipy.linalg import svd as ssvd
    utrue, strue, vtrue = ssvd(array(local))
    tol = 10e-04  # allow small error for iterative method
    assert allclose(svd.s[0], strue[0], atol=tol)
    assert allclose(vtest, vtrue[0, :], atol=tol) | allclose(-vtest, vtrue[0, :], atol=tol)
    assert allclose(utest, utrue[:, 0], atol=tol) | allclose(-utest, utrue[:, 0], atol=tol)


def test_svd_conversion():
    local = [
        array([1.0, 2.0, 6.0]),
        array([1.0, 3.0, 0.0]),
        array([1.0, 4.0, 6.0]),
        array([5.0, 1.0, 4.0])
    ]
    mat = fromlist(local)
    SVD(k=1, method='direct').calc(mat)


def test_ica():
    random.seed(42)
    data, s, a = ICA.make(shape=(100, 10), withparams=True)
    ica = ICA(c=2, svdmethod="direct", seed=1)
    ica.fit(data)
    s_ = array(ica.sigs.rows().collect())
    tol = 0.01
    assert allclose(abs(corrcoef(s[:, 0], s_[:, 0])[0, 1]), 1, atol=tol) or \
        allclose(abs(corrcoef(s[:, 0], s_[:, 1])[0, 1]), 1, atol=tol)
    assert allclose(abs(corrcoef(s[:, 1], s_[:, 0])[0, 1]), 1, atol=tol) or \
        allclose(abs(corrcoef(s[:, 1], s_[:, 1])[0, 1]), 1, atol=tol)
    assert allclose(array(data.values().collect()), dot(s_, ica.a.T))


def test_nmf_als():
    local = array([
        [1.0, 2.0, 6.0],
        [1.0, 3.0, 0.0],
        [1.0, 4.0, 6.0],
        [5.0, 1.0, 4.0]])
    mat = fromlist(local)
    h0 = array(
        [[0.09082617,  0.85490047,  0.57234593],
         [0.82766740,  0.21301186,  0.90913979]])
    htrue = array(
        [[0, 0.6010, 0.9163],
         [0.8970, 0.1556, 0.7423]])
    wtrue = array(
        [[4.5885, 1.5348],
         [1.3651, 0.2184],
         [5.9349, 1.0030],
         [0, 5.5147]])
    matscale = diag(norm(htrue, axis=1))
    htrue = dot(inv(matscale), htrue)
    wtrue = dot(wtrue, matscale)
    nmfthunder = NMF(k=2, method="als", h0=h0, maxiterations=9)
    nmfthunder.fit(mat)
    hthunder = nmfthunder.h
    wthunder = array(nmfthunder.w.values().collect())
    tol = 1e-03
    assert allclose(wthunder, wtrue, atol=tol)
    assert allclose(hthunder, htrue, atol=tol)


def test_nmf_init():
    local = array([
        [1.0, 2.0, 6.0],
        [1.0, 3.0, 0.0],
        [1.0, 4.0, 6.0],
        [5.0, 1.0, 4.0]])
    mat = fromlist(local)
    nmfthunder = NMF(k=2, history=True)
    nmfthunder.fit(mat)
    assert nmfthunder.error[-1] < 2.9950
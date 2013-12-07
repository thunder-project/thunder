# utilities for factorization

from numpy import random, mean, real, argsort, transpose, dot, inner, outer
from scipy.linalg import eig, inv, orth


# Direct method for computing SVD by calculating covariance matrix,
# only efficient when d is small
def svd1(data, k, meanSubtract=1):

    def outerSum(iterator):
        yield sum(outer(x, x) for x in iterator)

    n = data.count()

    if meanSubtract == 1:
        data = data.map(lambda x: x - mean(x))

    # TODO: confirm speed increase for mapPartitions vs map
    cov = data.mapPartitions(outerSum).reduce(lambda x, y: x + y) / n

    w, v = eig(cov)
    w = real(w)
    v = real(v)
    inds = argsort(w)[::-1]
    latent = w[inds[0:k]]
    comps = transpose(v[:, inds[0:k]])
    scores = data.map(lambda x: inner(x, comps))

    return comps, latent, scores


# ALS for computing SVD, preferable when d is large
def svd2(data, k, meanSubtract=1):

    def randomVector(ind, seed, k):
        random.seed(ind*100000*seed)
        r = random.rand(1, k)
        return r

    if meanSubtract == 1:
        data = data.map(lambda x: x - mean(x))

    v = data.map(lambda x, y: randomVector(x, 1, k))
    nIter = 5
    iteration = 1

    while iteration < nIter:
        # goal is to solve R = VU subject to U,V > 0
        # by iteratively updating U and V with least squares and clipping

        # precompute inv(V' * V)
        vinv = inv(v.map(lambda x: outer(x, x)).reduce(lambda x, y: (x+y)))

        # update U using least squares row-wise with inv(V' * V) * V * R (same as pinv(V) * R)
        u = data.join(v.map(lambda x: dot(vinv, x))).mapValues(lambda x, y: outer(x, y)).reduce(lambda x, y: x + y)

        # precompute pinv(U)
        uinv = transpose(inv(transpose(u)))

        # update V using least squares row-wise with R * pinv(U)
        v = data.mapValues(lambda x: dot(transpose(uinv), x))

        iteration += 1


def svd3(data, k, meanSubtract=1):

    n = data.count()
    d = len(data.first())

    if meanSubtract == 1:
        data = data.map(lambda x: x - mean(x))

    def outerSum(iterator):
        yield sum(outer(x, x) for x in iterator)

    C = random.rand(k, d)
    nIter = 20

    for iteration in range(nIter):
        Cold = C
        Cinv = dot(transpose(C), inv(dot(C, transpose(C))))
        XX = data.map(lambda x: dot(x, Cinv)).mapPartitions(outerSum).reduce(lambda x, y: x + y)
        XXinv = inv(XX)
        C = data.map(lambda x: outer(x, dot(dot(x, Cinv), XXinv))).reduce(
            lambda x, y: x + y)
        C = transpose(C)
        error = sum((C-Cold) ** 2)

    C = transpose(orth(transpose(C)))
    cov = data.map(lambda x: dot(x, transpose(C))).mapPartitions(outerSum).reduce(
        lambda x, y: x + y) / n
    w, v = eig(cov)
    w = real(w)
    v = real(v)
    inds = argsort(w)[::-1]
    latent = w[inds[0:k]]
    comps = dot(transpose(v[:, inds[0:k]]), C)
    scores = data.map(lambda x: inner(x, comps))

    return comps, latent, scores

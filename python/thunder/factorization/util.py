from numpy import *
from scipy.linalg import *

# Direct method for computing SVD by calculating covariance matrix,
# only efficient when d is small
def svd1(data,k,meanSubtract=1) :

    def outerSum(iterator) : yield sum(outer(x,x) for x in iterator)

    n = data.count()

    if meanSubtract == 1 :
        data = data.map(lambda x : x - mean(x))

    # TODO: test for speed increase vs straight map-reduce
    cov = data.mapPartitions(outerSum).reduce(lambda x,y : x + y) / n

    w, v = eig(cov)
    w = real(w)
    v = real(v)
    inds = argsort(w)[::-1]
    latent = w[inds[0:k]]
    comps = transpose(v[:,inds[0:k]])
    scores = data.map(lambda x : inner(x,comps))

    return comps, latent, scores

# ALS for computing SVD, preferable when d is large
def svd2(data,k,meanSubtract=1) :

    def randomVector(ind,seed,k) :
        random.seed(ind*100000*seed)
        r = random.rand(1,k)
        return r

    if meanSubtract == 1 :
        data = data.map(lambda x : x - mean(x))

    m = len(data.first()[1])

    v = data.map(lambda x,y : randomVector(x,seed1,k))
    u = random.randn(k,m)
    nIter = 5
    iIter = 1
    # fixed initialization for debugging
    #var u = factory2D.make(sc.textFile("data/h0.txt").map(parseLine _).toArray())
    #var w = sc.textFile("data/w0.txt").map(parseVector2 _)
    while (iter < nIter) :
        # goal is to solve R = VU subject to U,V > 0
        # by iteratively updating U and V with least squares and clipping
        
        # precompute inv(V' * V)
        vinv = inv(v.map(lambda x : outer(x,x)).reduce(lambda x,y : (x+y)))

        # update U using least squares row-wise with inv(V' * V) * V * R (same as pinv(V) * R)
        u = data.join(v.map(lambda x : dot(vinv,x))).mapValues(lambda x,y : outer(x,y)).reduce(lambda x,y: x + y)

        # precompute pinv(U)
        uinv = tranpose(inv(transpose(u)))

        # update V using least squares row-wise with R * pinv(U)
        v = data.mapValues(lambda x : dot(transpose(uinv),x))
        
        iter += 1

def svd3(data,k,meanSubtract=1) :

    if meanSubtract == 1 :
        data = data.map(lambda x : x - mean(x))

    C = random.rand(k,d)
    nIter = 20

    for iter in range(nIter):
        Cold = C
        Cinv = dot(transpose(C),inv(dot(C,transpose(C))))
        XX = data.map(
            lambda x : outerProd(dot(x,Cinv))).reduce(
            lambda x,y : x + y)
        XXinv = inv(XX)
        C = data.map(lambda x : outer(x,dot(dot(x,Cinv),XXinv))).reduce(
            lambda x,y: x + y)
        C = transpose(C)
        error = sum((C-Cold) ** 2)

    C = transpose(orth(transpose(C)))
    cov = data.map(lambda x : outerProd(dot(x,transpose(C)))).reduce(
        lambda x,y : x + y) / n
    w, v = eig(cov)
    w = real(w)
    v = real(v)
    inds = argsort(w)[::-1]
    latent = w[inds[0:k]]
    comps = dot(transpose(v[:,inds[0:k]]),C)
    scores = data.map(lambda x : inner(x,comps))

    return comps, latent, scores


# performs independent components analysis
#
# example:
# pyspark ica.py local data/sigs.txt raw results 4 4


import os
import argparse
from numpy import random, sqrt, zeros, real, dot, outer, diag, transpose, shape
from scipy.linalg import sqrtm, inv, orth
from thunder.util.dataio import parse, saveout
from thunder.factorization.util import svd1, svd2, svd3, svd4
from pyspark import SparkContext


def ica(data, k, c):

    n = data.count()

    # reduce dimensionality
    comps, latent, scores = svd4(data, k, 0)

    # whiten data
    whtMat = real(dot(inv(diag(sqrt(latent))), comps))
    unwhtMat = real(dot(transpose(comps), diag(sqrt(latent))))
    wht = data.map(lambda x: dot(whtMat, x))

    # do multiple independent component extraction
    B = orth(random.randn(k, c))
    Bold = zeros((k, c))
    iterNum = 0
    minAbsCos = 0
    tol = 0.000001
    iterMax = 1000
    errVec = zeros(iterMax)

    while (iterNum < iterMax) & ((1 - minAbsCos) > tol):
        iterNum += 1
        # update rule for pow3 nonlinearity (TODO: add other nonlins)
        B = wht.map(lambda x: outer(x, dot(x, B) ** 3)).sum() / n - 3 * B
        # orthognalize
        B = dot(B, real(sqrtm(inv(dot(transpose(B), B)))))
        # evaluate error
        minAbsCos = min(abs(diag(dot(transpose(B), Bold))))
        # store results
        Bold = B
        errVec[iterNum-1] = (1 - minAbsCos)

    # get unmixing matrix
    W = dot(transpose(B), whtMat)

    # get components
    sigs = data.map(lambda x: dot(W, x))

    return W, sigs, whtMat, unwhtMat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do independent components analysis")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("outputDir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("c", type=int)

    args = parser.parse_args()
    sc = SparkContext(args.master, "ica")
    lines = sc.textFile(args.dataFile)
    data = parse(lines, args.dataMode).cache()

    W, sigs, whtMat, unwhtMat = ica(data, args.k, args.c)

    outputDir = args.outputDir + "-ica"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    saveout(W, outputDir, "W", "matlab")
    saveout(sigs, outputDir, "sigs", "matlab", args.c)
    saveout(whtMat, outputDir, "whtMat", "matlab")
    saveout(unwhtMat, outputDir, "unwhtMat", "matlab")

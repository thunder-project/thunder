# performs robust PCA using ADMM
#
# TODO: Rewrite to avoid broadcast variables, make raw data an RDD,
#       S a sparse array, and we only keep the low rank representation of L
#
# example:
# pyspark rpca.py local data/rpca.txt raw results


import os
import argparse
from numpy import array, zeros, real, sqrt, sign, transpose, diag, outer, dot
from scipy.linalg import eig
from thunder.util.dataio import saveout, parse
from pyspark import SparkContext


def shrinkVec(x, thresh):
    tmp = abs(x)-thresh
    tmp[tmp < 0] = 0
    return tmp


def svdThreshold(RDD, thresh):
    cov = RDD.map(lambda x: outer(x, x)).reduce(lambda x, y: (x + y))
    w, v = eig(cov)
    sw = sqrt(w)
    inds = (sw-thresh) > 0
    d = (1/sw[inds]) * shrinkVec(sw[inds], thresh)
    vthresh = real(dot(dot(v[:, inds], diag(d)), transpose(v[:, inds])))
    return RDD.map(lambda x: dot(x, vthresh))


def shrinkage(RDD, thresh):
    return RDD.map(lambda x: sign(x) * shrinkVec(x, thresh))


def rpca(data):
    n = data.count()
    m = len(data.first())

    # create broadcast variables
    M = array(data.collect())
    L = zeros((n, m))
    S = zeros((n, m))
    Y = zeros((n, m))

    mu = float(12)
    lam = 1/sqrt(n)

    iterNum = 0
    iterMax = 50

    while iterNum < iterMax:
        iterNum += 1
        MSY = data.context.parallelize(M - S + (1/mu)*Y).cache()
        L = svdThreshold(MSY, 1/mu).collect()
        MLY = data.context.parallelize(M - L + (1/mu)*Y)
        S = shrinkage(MLY, lam/mu).collect()
        Y += mu * (M - L - S)

    return L, S

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do independent components analysis")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("outputDir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("c", type=int)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "rpca", pyFiles=egg)
    lines = sc.textFile(args.dataFile)
    data = parse(lines, args.dataMode).cache()

    L, S = rpca(data)

    outputDir = args.outputDir + "-rpca"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    saveout(L, outputDir, "L", "matlab")
    saveout(S, outputDir, "S", "matlab")
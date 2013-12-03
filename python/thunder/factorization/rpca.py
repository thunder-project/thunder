# usage: rpca <master> <dataFile> <outputDir>"
#
# performs robust PCA using ADMM
#
# TODO: Rewrite to avoid broadcast variables, make raw data an RDD,
#       S a sparse array, and we only keep the low rank representation of L
#
# example:
#
# pyspark rpca.py local data/rpca.txt results
#

import sys
import os
from numpy import array, zeros, real, sqrt, sign, transpose, diag, outer, dot
from scipy.linalg import eig
from thunder.util.dataio import saveout, parse
from pyspark import SparkContext

argsIn = sys.argv[1:]
if len(argsIn) < 3:
    print >> sys.stderr, "usage: rpca <master> <dataFile> <outputDir>"
    exit(-1)


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

# parse inputs
sc = SparkContext(argsIn[0], "rpca")
dataFile = str(argsIn[1])
outputDir = str(argsIn[2]) + "-rpca"
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# load data
lines = sc.textFile(dataFile)
data = parse(lines, "dff").cache()
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
    MSY = sc.parallelize(M - S + (1/mu)*Y).cache()
    L = svdThreshold(MSY, 1/mu).collect()
    MLY = sc.parallelize(M - L + (1/mu)*Y)
    S = shrinkage(MLY, lam/mu).collect()
    Y += mu * (M - L - S)

saveout(L, outputDir, "L", "matlab")
saveout(S, outputDir, "S", "matlab")

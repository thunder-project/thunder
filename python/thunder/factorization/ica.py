# ica <master> <inputFile> <outputFile> <k> <c>
#
# performs ICA
#
# k - number of principal components to use before ica
# c - number of independent components to find
#
# example:
#
# pyspark ica.py local data/sigs.txt results 4 4
#

import sys
import os
from numpy import random, sqrt, zeros, real, dot, outer, diag, transpose, shape
from scipy.linalg import sqrtm, inv, orth
from scipy.io import loadmat
from thunder.util.dataio import parse, saveout
from thunder.factorization.util import svd1, svd2, svd3, svd4
from pyspark import SparkContext
from pyspark.accumulators import AccumulatorParam


class MatrixAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return zeros(shape(value))

    def addInPlace(self, val1, val2):
        val1 += val2
        return val1

argsIn = sys.argv[1:]
if len(argsIn) < 5:
    print >> sys.stderr, "usage: ica <master> <inputFile> <outputFile> <k> <c>"
    exit(-1)

# parse inputs
sc = SparkContext(argsIn[0], "ica")
dataFile = str(argsIn[1])
outputDir = str(argsIn[2]) + "-ica"
k = int(argsIn[3])
c = int(argsIn[4])
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# load data
lines = sc.textFile(dataFile)
data = parse(lines, "raw", None, [150, 1000]).cache()
n = data.count()

# reduce dimensionality
comps, latent, scores = svd4(sc, data, k, 0)

# whiten data
#whtMat = real(dot(inv(diag(sqrt(latent))), comps))
#unwhtMat = real(dot(transpose(comps), diag(sqrt(latent))))

whtMat = loadmat(outputDir + "/whtMat.mat")['whtMat']
unwhtMat = loadmat(outputDir + "/unwhtMat.mat")['unwhtMat']
wht = data.map(lambda x: dot(whtMat, x))
#print(wht.first())

# save whitening matrices
#saveout(whtMat, outputDir, "whtMat", "matlab")
#saveout(unwhtMat, outputDir, "unwhtMat", "matlab")

# do multiple independent component extraction
B = orth(random.randn(k, c))
#B = loadmat(outputDir + "/B.mat")['B']
Bold = zeros((k, c))
iterNum = 0
minAbsCos = 0
tol = 0.000001
iterMax = 1000
errVec = zeros(iterMax)


global Bnew

def outerSum(x, y):
    global Bnew
    Bnew += outer(x, y)

while (iterNum < iterMax) & ((1 - minAbsCos) > tol):
    iterNum += 1
    # update rule for pow3 nonlinearity (TODO: add other nonlins)
    #B = wht.map(lambda x: (x, dot(x, B) ** 3)).mapPartitions(outerSum).reduce(lambda x, y: x + y) / n - 3 * B
    Bnew = sc.accumulator(zeros((k, c)), MatrixAccumulatorParam())
    wht.map(lambda x: (x, dot(x, B) ** 3)).foreach(lambda x: outerSum(x[0], x[1]))
    B = Bnew.value / n - 3 * B
    #B = wht.map(lambda x: outer(x, dot(x, B) ** 3)).reduce(lambda x, y: x + y) / n - 3 * B
    # orthognalize
    B = dot(B, real(sqrtm(inv(dot(transpose(B), B)))))
    # evaluate error
    minAbsCos = min(abs(diag(dot(transpose(B), Bold))))
    # store results
    Bold = B
    saveout(B, outputDir, "B", "matlab")
    errVec[iterNum-1] = (1 - minAbsCos)

# get unmixing matrix
W = dot(transpose(B), whtMat)

# get components
sigs = data.map(lambda x: dot(W, x))

# save output files
saveout(W, outputDir, "W", "matlab")
saveout(sigs, outputDir, "sigs", "matlab", c)

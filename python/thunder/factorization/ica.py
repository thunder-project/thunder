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
from numpy import random, sqrt, zeros, real, dot, outer, transpose
from scipy.linalg import diag, sqrtm, inv, orth
from thunder.util.dataio import parse, saveout
from thunder.factorization.util import svd1
from pyspark import SparkContext

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
data = parse(lines, "raw").cache()
n = data.count()

# reduce dimensionality
comps, latent, scores = svd1(data, k, 0)

# whiten data
whtMat = real(dot(inv(diag(sqrt(latent))), comps))
unwhtMat = real(dot(transpose(comps), diag(sqrt(latent))))
wht = data.map(lambda x: dot(whtMat, x)).cache()

# do multiple independent component extraction
B = orth(random.randn(k, c))
Bold = zeros((k, c))
iterNum = 0
minAbsCos = 0
termTol = 0.000001
iterMax = 1000
errVec = zeros(iterMax)

while (iterNum < iterMax) & ((1 - minAbsCos) > termTol):
    iterNum += 1
    # update rule for pow3 nonlinearity (TODO: add other nonlins)
    B = wht.map(lambda x: outer(x, dot(x, B) ** 3)).reduce(lambda x, y: x + y) / n - 3 * B
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

# save output files
saveout(W, outputDir, "W", "matlab")
saveout(sigs, outputDir, "sigs", "matlab", c)

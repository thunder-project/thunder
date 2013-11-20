# ica <master> <inputFile> <outputFile> <k> <c>
# 
# perform ica
#

import sys
import os
from numpy import *
from scipy.linalg import *
from thunder.util.dataio import *
from thunder.factorization.util import *
from pyspark import SparkContext

argsIn = sys.argv[1:]
if len(sys.argv) < 5:
  print >> sys.stderr, \
  "(ica) usage: ica <master> <inputFile> <outputFile> <k> <c>"
  exit(-1)

# parse inputs
sc = SparkContext(sys.argv[1], "ica")
dataFile = str(sys.argv[2])
outputDir = str(sys.argv[3]) + "-ica"
k = int(sys.argv[4])
c = int(sys.argv[5])
if not os.path.exists(outputDir) : os.makedirs(outputDir)

# load data
lines = sc.textFile(dataFile)
data = parse(lines, "dff").cache()
n = data.count()

# reduce dimensionality
comps, latent, scores = svd1(data,k,0)

# whiten data
whtMat = real(dot(inv(diag(sqrt(latent))),comps))
unwhtMat = real(dot(transpose(comps),diag(sqrt(latent))))
wht = data.map(lambda x : dot(whtMat,x))

# do multiple independent component extraction
B = orth(random.randn(k,c))
Bold = zeros((k,c))
iterNum = 0
minAbsCos = 0
termTol = 0.000001
iterMax = 1000
errVec = zeros(iterMax)

while (iterNum < iterMax) & ((1 - minAbsCos) > termTol):
	iterNum += 1
	# update rule for pow3 nonlinearity (TODO: add other nonlins)
	B = wht.map(lambda x : outer(x,dot(x,B) ** 3)).reduce(lambda x,y : x + y) / n - 3 * B
	# orthognalize
	B = dot(B,real(sqrtm(inv(dot(transpose(B),B)))))
	# evaluate error
	minAbsCos = min(abs(diag(dot(transpose(B),Bold))))
	# store results
	Bold = B
	errVec[iterNum-1] = (1 - minAbsCos)

# get unmixing matrix
W = dot(transpose(B),whtMat)

# get components
sigs = data.map(lambda x : dot(W,x))

# save output files
saveout(W,outputDir,"W","matlab")
saveout(sigs,outputDir,"sigs","matlab")




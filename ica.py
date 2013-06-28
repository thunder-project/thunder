# ica <master> <inputFile> <outputFile_Components> <outputFile_Weights>
# 
# Perform ICA on a data matrix.
# Input is a local text file or a file in HDFS.
# Format should be rows of ' ' separated values.
# - Example: space (rows) x time (cols).
# - Rows should be whichever dim is larger.
# Returns independent components and unmixing matrix in separate files

# k is number of PCs to use in initial dimensionality reduction
# c is the number of ICA components to return

import sys
from numpy import *
from scipy.linalg import *
from pyspark import SparkContext

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(ica) usage: ica <master> <inputFile> <outputFile> <k> <c>"
  exit(-1)

def parseVector(line):
    return array([float(x) for x in line.split(' ')])

# parse inputs
sc = SparkContext(sys.argv[1], "ica")
lines = sc.textFile(sys.argv[2])
outputFile = str(sys.argv[3])
k = int(sys.argv[4])
c = int(sys.argv[5])

# compute covariance matrix
data = lines.map(parseVector).cache()
n = data.count()
m = len(data.first())
meanVec = data.reduce(lambda x,y : x+y) / n
sub = data.map(lambda x : x - meanVec)
cov = sub.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / n

# do eigenvector decomposition
w, v = eig(cov)
inds = argsort(w)[::1]
kEigVecs = v[:,inds[0:k]]
kEigVals = w[inds[0:k]]

# whiten data
whtMat = real(dot(inv(sqrtm(diag(kEigVals))),transpose(kEigVecs)))
unwhtMat = dot(kEigVecs,sqrtm(diag(kEigVals)))
wht = sub.map(lambda x : dot(whtMat,x))

# do multiple independent component extraction
B = orth(random.randn(k,c))
Bold = zeros((k,c))
iterNum = 0
minAbsCos = 0
termTol = 0.0001
iterMax = 1000
errVec = zeros(iterMax)

while (iterNum < iterMax) & ((1 - minAbsCos) > termTol):
	iterNum += 1
	print "(ica) starting iteration " + str(iterNum)
	# update rule for pow3 nonlinearity (todo: add other nonlins)
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

# get unmixed signals
sigs = sub.map(lambda x : dot(W,x)).collect()

# save output files
print("(ica) writing output...")
savetxt("out-W-"+outputFile+".txt",W,fmt='%.8f')
savetxt("out-sigs-"+outputFile+".txt",sigs,fmt='%.8f')




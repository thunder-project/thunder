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
from numpy.linalg import *
from pyspark import SparkContext

if len(sys.argv) < 5
  print >> sys.stderr, \
  "(ica) usage: ica <master> <inputFile> <k> <c> <outputFile_Components> <outputFile_Weights>"
  exit(-1)

def parseVector(line):
    return array([float(x) for x in line.split(' ')])

# parse inputs
sc = SparkContext(sys.argv[1], "ica")
lines = sc.textFile(sys.argv[2])
k = int(sys.argv[3])
c = int(sys.argv[4])
convergeDist = float(sys.argv[5])
fileOutComp = str(sys.argv[6])
fileOutWeight = str(sys.argv[7])

# compute covariance matrix
data = lines.map(parseVector).cache()
n = data.count()
m = len(data.first())
meanVec = data.reduce(lambda x,y : x+y) / n
sub = data.map(lambda x : x - meanVec)
cov = sub.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / (n-1)

# do eigenvector decomposition
w, v = eig(cov)
inds = argsort(w)[::1]
kEigVecs = v[:,inds[0:k]]
kEigVals = w[inds[0:k]]

# whiten data
whtMat = dot(inv(diag(kEigVals) ** 0.5),transpose(kEigVecs))
wht = sub.map(lambda x : dot(whtMat,x))

# do multiple independent component extraction
W = random.randn(c,k)
Wold = zeros((c,k))
iterNum = 0
minAbsCos = 0
termTol = 1e-6
iterMax = 100
errVec = zeros(iterMax)

while (iterNum < iterMax) & ((1 - minAbsCos)>termTol)
iterNum += 1
W = wht.map(lambda x : outer(x,dot(x,W) ** 3)).reduce(lambda x,y : x + y) / n - 3 * W
W = W * real(complex_(inv(dot(transpose(W),W))) ** 0.5)
minAbsCos = min(abs(diag(dot(W,Wold))))
Wold = W
errVec[iterNum-1] = (1 - minAbsCos)

# need to add code to save output filesex




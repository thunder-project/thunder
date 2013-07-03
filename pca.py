# pca <master> <inputFile> <outputFile> <slices> <k> <dim>
# 
# performs pca on a data matrix
# input is a local text file or a file in hdfs 
# format should be rows of ' ' separated values
# - example: space (rows) x time (cols)
# - rows should be whichever dim is larger
# 'dim' is dimension to subtract mean along
# 'k' is number of pcs to return
# writes pcs (in both dims) and eigenvalues to text

import sys
import os
from numpy import *
from scipy.linalg import *
from pyspark import SparkContext

if len(sys.argv) < 7:
  print >> sys.stderr, \
    "(pca) usage: pca <master> <inputFile> <outputFile> <slices> <k> <dim>"
  exit(-1)

def parseVector(line):
    return array([float(x) for x in line.split(' ')])

# parse inputs
sc = SparkContext(sys.argv[1], "pca")
inputFile = str(sys.argv[2])
outputFile = str(sys.argv[3])
slices = int(sys.argv[4])
k = int(sys.argv[5])
dim = int(sys.argv[6])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)

# load data
lines = sc.textFile(inputFile,slices)
data = lines.map(parseVector).cache()
n = data.count()

# do mean subtraction
if dim==1:
	meanVec = data.reduce(lambda x,y : x+y) / n
	sub = data.map(lambda x : x - meanVec)
elif dim==2:
	meanVec = data.reduce(lambda x,y : x+y) / n
	sub = data.map(lambda x : x - np.mean(x))
else:
 print >> sys.stderr, \
 "(pca) dim must be 1 or 2"
 exit(-1)

# do eigendecomposition
cov = sub.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / (n - 1)
w, v = eig(cov)
inds = argsort(w)[::-1]
sortedDim2 = transpose(v[:,inds[0:k]])
latent = w[inds[0:k]]

print("(pca) writing output...")
np.savetxt(outputFile+"/"+"out-dim2-"+outputFile+".txt",sortedDim2,fmt='%.8f')
np.savetxt(outputFile+"/"+"out-latent-"+outputFile+".txt",latent,fmt='%.8f')
for ik in range(0,k):
	sortedDim1 = sub.map(lambda x : inner(x,sortedDim2[ik,:]))
	np.savetxt(outputFile+"/"+"out-dim1-"+str(ik)+"-"-outputFile+".txt",sortedDim1.collect(),fmt='%.8f')



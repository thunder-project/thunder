# pca <master> <inputFile> <k> <outputFile>
# 
# performs pca on a data matrix
# input is a local text file or a file in hdfs 
# format should be rows of ' ' separated values
# - example: space (rows) x time (cols)
# - rows should be whichever dim is larger
# subtracts means along dimension 'dim'
# writes 'k' pcs (in both dims) and eigenvalues to text

import sys

import numpy as np
from numpy import linalg as la
from pyspark import SparkContext

if len(sys.argv) < 6:
  print >> sys.stderr, \
    "(pca) usage: pca <master> <inputFile> <outputFile> <k> <dim>"
  exit(-1)

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def outerProd(vec): 
	return np.outer(vec,vec)

sc = SparkContext(sys.argv[1], "pca")
lines = sc.textFile(sys.argv[2])
fileOut = str(sys.argv[3])
k = int(sys.argv[4])
dim = int(sys.argv[5])

data = lines.map(parseVector).cache()
n = data.count()

if dim==1:
	meanVec = data.reduce(lambda x,y : x+y) / n
	sub = data.map(lambda x : x - meanVec)
elif dim==2:
	meanVec = data.reduce(lambda x,y : x+y) / n
	sub = data.map(lambda x : x - np.mean(x))
else:
 print >> sys.stderr, \
 "(pca) dim must be 1 or 2"
 exist(-1)

cov = sub.map(outerProd).reduce(lambda x,y : (x + y)) / (n - 1)
w, v = la.eig(cov)
inds = np.argsort(w)[::-1]
sortedDim2 = v[:,inds[0:k]].transpose()
sortedDim1 = sub.map(lambda x : np.inner(x,sortedDim2))
latent = w[inds[0:k]]

print("(pca) writing output...")
np.savetxt("out-dim1-"+fileOut+".txt",sortedDim1.collect(),fmt='%.8f')
np.savetxt("out-dim2-"+fileOut+".txt",sortedDim2,fmt='%.8f')
np.savetxt("out-latent-"+fileOut+".txt",latent,fmt='%.8f')





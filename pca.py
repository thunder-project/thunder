# pca <master> <inputFile> <k> <outputFile>
# 
# - performs pca on a data matrix efficiently
# when the number of observations is fewer
# than the number of dimensions
# - each row is a dimension
# - subtracts off the mean of each dimension
# - returns k principal components for the k largest eigenvalues,
# as well as the corresponding k eigenvectors ("scores")
# 


import sys

import numpy as np
from numpy import linalg as la
from pyspark import SparkContext

if len(sys.argv) < 4:
  print >> sys.stderr, \
    "Usage: pca <master> <inputFile> <k> <outputFile>"
  exit(-1)

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def outerProd(vec): 
	return np.outer(vec,vec)

sc = SparkContext("spark://"+sys.argv[1]+":7077", "pca")
lines = sc.textFile(sys.argv[2])
k = int(sys.argv[3])
fileOut = str(sys.argv[4])

n = data.count()
meanVec = data.reduce(lambda x,y : x+y) / n
sub = data.map(lambda x : x - meanVec)
cov = sub.map(outerProd).reduce(lambda x,y : (x + y)) / (n-1)
w, v = la.eig(cov)
inds = np.argsort(w)[::-1]
sortedEigVecs = v[:,inds[0:k]].transpose()
sortedEigVals = w[inds[0:k]]

print("(pca) writing output...")
np.savetxt("pca-output/out-time-"+fileOut+".txt",sortedEigVecs,fmt='%.8f')
np.savetxt("pca-output/out-eigs-"+fileOut+".txt",sortedEigVals,fmt='%.8f')
np.savetxt("pca-output/out-cov-"+fileOut+".txt",cov,fmt='%.8f')

for num in range(0,k):
	princomp = sub.map(lambda x : np.inner(x,sortedEigVecs[num]))
	princomp.saveAsTextFile("hdfs://"+sys.argv[1]+":9000"+"/pca-output/out-space-"+str(num)+"-"+fileOut)




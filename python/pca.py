# pca <master> <inputFile> <outputFile> <k> <dim>
# 
# performs pca on a data matrix
# each row is (x,y,z,timeseries)
#

import sys
import os
from numpy import *
from scipy.linalg import *
import logging
from pyspark import SparkContext

if len(sys.argv) < 6:
  print >> sys.stderr, \
    "(pca) usage: pca <master> <inputFile> <outputFile> <k> <dim>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	med = median(ts)
	ts = (ts - med) / (med + 0.1) # convert to dff
	return ts
	
# parse inputs
sc = SparkContext(sys.argv[1], "pca")
inputFile = str(sys.argv[2])
k = int(sys.argv[4])
outputFile = str(sys.argv[3]) + "-pca-pcs-" + str(k)
dim = int(sys.argv[5])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# load data
lines = sc.textFile(inputFile)
data = lines.map(parseVector).cache()
n = data.count()

# do mean subtraction
if dim==1:
	meanVec = data.reduce(lambda x,y : x+y) / n
	sub = data.map(lambda x : x - meanVec)
elif dim==2:
	meanVec = data.reduce(lambda x,y : x+y) / n
	sub = data.map(lambda x : x - mean(x))
else:
 print >> sys.stderr, \
 "(pca) dim must be 1 or 2"
 exit(-1)

# do eigendecomposition
logging.info("(pca) computing covariance")
cov = sub.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / n
logging.info("(pca) doing eigendecomposition")
w, v = eig(cov)
w = real(w)
v = real(v)
inds = argsort(w)[::-1]
sortedDim2 = transpose(v[:,inds[0:k]])
latent = w[inds[0:k]]

logging.info("(pca) writing output...")
savetxt(outputFile+"/"+"evecs.txt",sortedDim2,fmt='%.8f')
savetxt(outputFile+"/"+"evals.txt",latent,fmt='%.8f')
for ik in range(0,k):
	out = sub.map(lambda x : inner(x,sortedDim2[ik,:]))
	savetxt(outputFile+"/"+"scores-"+str(ik)+".txt",out.collect(),fmt='%.4f')
	
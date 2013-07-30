import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(stc) usage: stc <master> <inputFile_A> <inputFile_y> <outputFile> <mxLag>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[1:])
	ts = (ts - mean(ts))/std(ts)
	return ts

# parse inputs
sc = SparkContext(sys.argv[1], "stc")
inputFile_A = str(sys.argv[2])
inputFile_y = str(sys.argv[3])
mxLag = int(sys.argv[5])
outputFile = str(sys.argv[4]) + "-stc"
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(stc) loading data")
lines_A = sc.textFile(inputFile_A)
lines_y = sc.textFile(inputFile_y)
y = array([float(x) for x in lines_y.collect()[0].split(' ')])
A = lines_A.map(parseVector)
n = A.count()

# subtract the mean and cache
logging.info("(stc) subtracting mean")
meanVec = A.reduce(lambda x,y : x+y) / n
sub = A.map(lambda x : x - meanVec).cache()

# compute raw covariance
logging.info('(stc) computing covariance')
covRaw = sub.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / n

# compute triggered covariance
logging.info('(stc) computing triggered covariance')
covTrig = sub.map(lambda x : outer(x*y,x*y)).reduce(lambda x,y : (x + y)) / n

# save results
savemat(outputFile+"/"+"stc-covRaw.mat",mdict={'covRaw':covRaw},do_compression='true')
savemat(outputFile+"/"+"stc-covTrig.mat",mdict={'covTrig':covTrig},do_compression='true')

# do eigenvector decomposition of raw covariance
# logging.info('(stc) doing eigendecomposition')
# w, v = eig(cov)
# inds = argsort(w)[::-1]
# kEigVecs = v[:,inds[0:k]]
# kEigVals = w[inds[0:k]]
# whtMat = real(dot(inv(diag(sqrt(kEigVals))),transpose(kEigVecs)))

# # solve generalized eigenvector problem
# wTrig, vTrig = eig(dot(dot(whtMat,covTrig),transpose(whtMat)))
# inds = argsort(wTrig)[::-1]
# kEigVecsTrig = transpose(vTrig[:,inds[0:k]])
# kEigValsTrig = wTrig[inds[0:k]]

#out = sub.map(lambda x : inner(x,kEigVecsTrig[ik,:]))
import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(lowdim) usage: lowdim <master> <inputFile_X> <inputFile_y> <mode> <outputFile> <k>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	med = median(ts)
	ts = (ts - med) / (med + 0.1) # convert to dff
	return ts

def xcorr(x,y):
	x1 = 
	return ts

# parse inputs
sc = SparkContext(sys.argv[1], "lowdim")
inputFile_X = str(sys.argv[2])
inputFile_y = str(sys.argv[3])
mode = str(sys.argv[4])
outputFile = str(sys.argv[5]) + "-lowdim"
k = int(sys.argv[6])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(lowdim) loading data")
lines_X = sc.textFile(inputFile_X) # the data
X = lines_X.map(parseVector).cache()
y = loadmat(inputFile_y)['y']
if mode == 'mean' :
	resp = X.map(lambda x : dot(y,x))
if mode == 'regress' : 
	resp = X.map(lambda x : dot(y,(x-mean(x))/norm(x)))

# compute covariance
logging.info("(lowdim) getting count")
n = resp.count()
logging.info("(lowdim) computing covariance")
cov = resp.map(lambda x : outer(x-mean(x),x-mean(x))).reduce(lambda x,y : (x + y)) / n

logging.info("(lowdim) doing eigendecomposition")
w, v = eig(cov)
w = real(w)
v = real(v)
inds = argsort(w)[::-1]
sortedDim2 = transpose(v[:,inds[0:k]])
latent = w[inds[0:k]]

logging.info("(lowdim) writing evecs and evals")
savemat(outputFile+"/"+"evecs.mat",mdict={'evecs':sortedDim2},oned_as='column',do_compression='true')
savemat(outputFile+"/"+"evals.mat",mdict={'evals':latent},oned_as='column',do_compression='true')

for ik in range(0,k):
	logging.info("(lowdim) writing scores for pc " + str(ik))
	#out = X.map(lambda x : float16(inner(dot(y,x) - mean(dot(y,x)),sortedDim2[ik,:])))
	out = X.map(lambda x : float16(inner(dot(y,(x-mean(x))/norm(x)) - mean(dot(y,(x-mean(x))/norm(x))),sortedDim2[ik,:])))
	savemat(outputFile+"/"+"scores-"+str(ik)+".mat",mdict={'scores':out.collect()},oned_as='column',do_compression='true')

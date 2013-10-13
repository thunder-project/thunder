# lowdimPair <master> <inputFile_X1> <inputFile_X2> <inputFile_y1> <mode> <outputFile> <k>"
# 
# perform dimensionality reduction on data set 1
# then examine responses from data set 2 in the recovered subspace
# return low-dimensional subspace, as well as raw time
# series projected into that space
# each row is (x,y,z,timeseries)
#

import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(lowdimPair) usage: lowdimPair <master> <inputFile_X1> <inputFile_X2> <inputFile_y1> <mode> <outputFile> <k>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	meanVal = mean(ts)
	ts = (ts - meanVal) / (meanVal + 0.1) # convert to dff
	#ind = int(vec[0]) + int((vec[1] - 1)*2048) + int((vec[2] - 1)*1364*2048)
	return ((int(vec[0]),int(vec[1]),int(vec[2])),ts) # (x,y,z),(tseries) pair 

sc = SparkContext(sys.argv[1], "lowdimPair")
inputFile_X1 = str(sys.argv[2])
inputFile_X2 = str(sys.argv[3])
inputFile_y = str(sys.argv[4])
mode = str(sys.argv[5])
outputFile = str(sys.argv[6]) + "-lowdimPair"
k = int(sys.argv[7])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

logging.info("(lowdimPair) loading data")
lines_X1 = sc.textFile(inputFile_X1) # the first data set
X1 = lines_X1.map(parseVector).cache()
lines_X2 = sc.textFile(inputFile_X2) # the first data set
X2 = lines_X2.map(parseVector)
y = loadmat(inputFile_y)['y']
y = y.astype(float)

if mode == 'mean' :
	resp = X1.mapValues(lambda x : dot(y,x))

# compute covariance
logging.info("(lowdimPair) getting count")
n = resp.count()
logging.info("(lowdimPair) computing covariance")
cov = resp.map(lambda (k,x) : outer(x-mean(x),x-mean(x))).reduce(lambda x,y : (x + y)) / n

logging.info("(lowdimPair) doing eigendecomposition")
w, v = eig(cov)
w = real(w)
v = real(v)
inds = argsort(w)[::-1]
sortedDim2 = transpose(v[:,inds[0:k]])
latent = w[inds[0:k]]

for ik in range(0,k):
	logging.info("(lowdimPair) writing trajectories")
	scores = resp.mapValues(lambda x : inner(x - mean(x),sortedDim2[ik,:]))
	traj = X2.join(scores).map(lambda (k,x) : x[0] * x[1]).reduce(lambda x,y : x+y)
	savemat(outputFile+"/"+"traj-"+str(ik)+".mat",mdict={'traj':traj},oned_as='column',do_compression='true')





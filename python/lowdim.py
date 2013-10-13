# lowdim <master> <inputFile_X> <inputFile_y> <mode> <outputFile> <k>"
# 
# perform two stages of dimensionality reduction
# first reduce each time series using the specified method
# then do PCA
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
  "(lowdim) usage: lowdim <master> <inputFile_X> <inputFile_y> <mode> <outputFile> <k>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	meanVal = mean(ts)
	ts = (ts - meanVal) / (meanVal + 0.1) # convert to dff
	return ts

def clip(vec,val):
	vec[vec<val] = val
	return vec

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
y = y.astype(float)

if mode == 'mean' :
	resp = X.map(lambda x : dot(y,x))
if mode == 'standardize' :
	resp = X.map(lambda x : dot(y,(x-mean(x))/norm(x)))
if mode == 'regress' : 
	yhat = dot(inv(dot(y,transpose(y))),y)
	resp = X.map(lambda x : dot(yhat,x)[1:])
	r2 = X.map(lambda x : 1.0 - sum((dot(transpose(y),dot(yhat,x)) - x) ** 2) / sum((x - mean(x)) ** 2)).collect()
	savemat(outputFile+"/"+"r2.mat",mdict={'r2':r2},oned_as='column',do_compression='true')
	#vals = array([0,2,4,6,8,10,12,14,16,20,25,30])
	vals = array([2.5,7.5,12.5,17.5,22.5,27.5,32.5,37.5,42.5,47.5])
	tuning = resp.map(lambda x : clip(x,0)).map(lambda x : x / sum(x)).map(lambda x : dot(x,vals)).collect()
	savemat(outputFile+"/"+"tuning.mat",mdict={'tuning':tuning},oned_as='column',do_compression='true')

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
savemat(outputFile+"/"+"cov.mat",mdict={'cov':cov},oned_as='column',do_compression='true')
savemat(outputFile+"/"+"evecs.mat",mdict={'evecs':sortedDim2},oned_as='column',do_compression='true')
savemat(outputFile+"/"+"evals.mat",mdict={'evals':latent},oned_as='column',do_compression='true')

for ik in range(0,k):
	logging.info("(lowdim) writing trajectories for pc " + str(ik))
	traj = X.map(lambda x : x * inner(dot(y,x) - mean(dot(y,x)),sortedDim2[ik,:]) ).reduce(lambda x,y : x + y)
	savemat(outputFile+"/"+"traj-"+str(ik)+".mat",mdict={'traj':traj},oned_as='column',do_compression='true')

for ik in range(0,k):
	logging.info("(lowdim) writing scores for pc " + str(ik))
	#out = X.map(lambda x : float16(inner(dot(y,x) - mean(dot(y,x)),sortedDim2[ik,:])))
	#out = X.map(lambda x : float16(inner(dot(y,(x-mean(x))/norm(x)) - mean(dot(y,(x-mean(x))/norm(x))),sortedDim2[ik,:])))
	out = resp.map(lambda x : float16(inner(x - mean(x),sortedDim2[ik,:])))
	savemat(outputFile+"/"+"scores-"+str(ik)+".mat",mdict={'scores':out.collect()},oned_as='column',do_compression='true')





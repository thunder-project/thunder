# regress <master> <inputFile_Y> <inputFile_X> <outputFile> <analMode> <outputMode>
# 
# time series regression on a data matrix
# each row is (x,y,z,timeseries)
# inputs are signals to regress against
# can process results either by doing dimensionality reduction
# or by fitting a parametric model
#

import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

argsIn = sys.argv[1:]

if len(argsIn) < 7:
  print >> sys.stderr, \
  "(regress) usage: regress <master> <inputFile_Y> <inputFile_X> <outputFile> <analMode> <outputMode>"
  exit(-1)

def parseVector(line,mode="raw",xyz=0,inds=None):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	if inds is not None :
		ts = ts[inds[0]:inds[1]]
	if mode == "dff" :
		meanVal = mean(ts)
		ts = (ts - meanVal) / (meanVal + 0.1)
	if xyz == 1 :
		return ((int(vec[0]),int(vec[1]),int(vec[2])),ts)
	else :
		return ts

def getRegression(y,model) :
	if model.regressMode == 'mean' :
		return dot(model.X,y)
	if model.regressMode == 'linear' :
		return dot(model.Xhat,y)[1:]
	if model.regressMode == 'bilinear' :
		return dot(model.Xhat,y)

# parse inputs
sc = SparkContext(argsIn[0], "regress")
inputFile_Y = str(argsIn[1])
inputFile_X = str(argsIn[2])
outputFile = str(argsIn[3]) + "-regress"
regressMode = str(argsIn[4])
outputMode = str(argsIn[5])

if not os.path.exists(outputFile) :
	os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(regress) loading data")
lines_Y = sc.textFile(inputFile_Y)

# parse model
class model : pass
model.regressMode = regressMode
if regressMode == 'mean' :
	X = loadmat(inputFile_X + "_X.mat")['X']
	X = X.astype(float)
	model.X = X
if regressMode == 'linear' :
	X = loadmat(inputFile_X + "_X.mat")['X']
	X = X.astype(float)
	Xhat = dot(inv(dot(X,transpose(X))),X)
	model.X = X
	model.Xhat = Xhat
if regressMode == 'bilinear' :
	X1 = loadmat(inputFile_X + "_X1.mat")['X1']
	X2 = loadmat(inputFile_X + "_X2.mat")['X2']
	X1hat = dot(inv(dot(X1,transpose(X1))),X1)
	X2hat = dot(inv(dot(X2,transpose(X2))),X2)
	model.X1 = X1
	model.X2 = X2
	model.X1hat = X1hat
	model.X2hat = X2hat
if outputMode == 'tuning' :
	s = loadmat(inputFile_X + "_s.mat")['s']
	model.s = s

# compute parameter estimates
B = Y.map(lambda y : getRegression(y,model))

# process outputs using pca
if outputMode == 'pca' :
	k = 3
	n = B.count()
	cov = B.map(lambda b : outer(b-mean(b),b-mean(b))).reduce(lambda x,y : (x + y)) / n
	w, v = eig(cov)
	w = real(w)
	v = real(v)
	inds = argsort(w)[::-1]
	comps = transpose(v[:,inds[0:k]])
	savemat(outputFile+"/"+"comps.mat",mdict={'comps':comps},oned_as='column',do_compression='true')
	latent = w[inds[0:k]]
	for ik in range(0,k) :
		scores = Y.map(lambda y : float16(inner(getRegression(y,model) - mean(getRegression(y,model)),comps[ik,:])))
		savemat(outputFile+"/"+"scores-"+str(ik)+".mat",mdict={'scores':scores.collect()},oned_as='column',do_compression='true')
	traj = Y.map(lambda x : outer(x,inner(getRegression(y,model) - mean(getRegression(y,model)),comps))).reduce(lambda x,y : x + y) / n
	savemat(outputFile+"/"+"traj.mat",mdict={'traj':traj},oned_as='column',do_compression='true')

# process output with a parametric tuning curve
#if outputMode == 'tuning' :
























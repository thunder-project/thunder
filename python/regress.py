# regress <master> <inputFile_Y> <inputFile_x1> <inputFile_x2> <outputFile> <mxLag>
# 
# time series regression on a data matrix
# each row is (x,y,z,timeseries)
# inputs are signals to regress against
#
# TODO: add options for number of regressors and interactions
#

import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 7:
  print >> sys.stderr, \
  "(regress) usage: regress <master> <inputFile_Y> <inputFile_x1> <inputFile_x2> <outputFile> <mxLag>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	med = median(ts)
	ts = (ts - med) / (med + 0.1) # convert to dff
	return ts

def getRegStats(y,pinvX):
	y = y[hstack((arange(0,93),arange(130,259)))]
	betaest = dot(transpose(y),pinvX)
	predic = dot(betaest,X)
	SSE = sum((predic - y) ** 2)
	SST = sum((y - mean(y)) ** 2)
	r2 = 1 - SSE/SST
	return hstack((betaest,r2))

# parse inputs
sc = SparkContext(sys.argv[1], "regress")
inputFile_Y = str(sys.argv[2])
inputFile_x1 = str(sys.argv[3])
inputFile_x2 = str(sys.argv[4])
mxLag = int(sys.argv[6])
outputFile = str(sys.argv[5]) + "-regress-mxLag-" + str(mxLag)
lags = arange(2*mxLag) - floor(2*mxLag/2)
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(regress) loading data")
lines_Y = sc.textFile(inputFile_Y)
lines_x1 = sc.textFile(inputFile_x1)
lines_x2 = sc.textFile(inputFile_x2)
x1 = array([float(x) for x in lines_x1.collect()[0].split(' ')])
x2 = array([float(x) for x in lines_x2.collect()[0].split(' ')])
Y = lines_Y.map(parseVector).cache()
n = len(x1)

# compute sta
for lag in lags:
	logging.info('(regress) doing regression with time lag ' + str(lag))
	X = vstack((ones((n,)),roll(x1,int(lag)),roll(x2,int(lag)),roll(x1*x2,int(lag))))
	X = X[:,hstack((arange(0,93),arange(130,259)))]
	Xpinv = pinv(X)
	betas = Y.map(lambda y : getRegStats(y,Xpinv))
	logging.info('(regress) saving results...')
	nm = str(int(lag))
	if (lag < 0):
		nm = "n" + nm[1:]
	savemat(outputFile+"/"+"b1-lag-"+nm+".mat",mdict={'b1':betas.map(lambda x : x[1]).collect()},oned_as='column',do_compression='true')
	savemat(outputFile+"/"+"b2-lag-"+nm+".mat",mdict={'b2':betas.map(lambda x : x[2]).collect()},oned_as='column',do_compression='true')
	savemat(outputFile+"/"+"b12-lag-"+nm+".mat",mdict={'b12':betas.map(lambda x : x[3]).collect()},oned_as='column',do_compression='true')
	savemat(outputFile+"/"+"r2-lag-"+nm+".mat",mdict={'r2':betas.map(lambda x : x[4]).collect()},oned_as='column',do_compression='true')

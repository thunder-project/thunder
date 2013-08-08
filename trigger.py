import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 7:
  print >> sys.stderr, \
  "(trigger) usage: trigger <master> <inputFile_Y> <inputFile_x1> <inputFile_x2> <outputFile> <mxFrame>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[1:])
	return ts

def getResp(y,X,lag):
	X = X[:,hstack((arange(0,93),arange(130,259)))]
	y = y[hstack((arange(0,93),arange(130,259)))]
	inds1 = (X[0,:] == 1) & (X[1,:] == lag)
	inds2 = (X[0,:] == 0) & (X[1,:] == lag)
	return mean(y[inds1]) - mean(y[inds2])

# parse inputs
sc = SparkContext(sys.argv[1], "trigger")
inputFile_Y = str(sys.argv[2])
inputFile_x1 = str(sys.argv[3])
inputFile_x2 = str(sys.argv[4])
mxFrame = int(sys.argv[6])
outputFile = str(sys.argv[5]) + "-trigger-frames-" + str(mxFrame)
lags = arange(mxFrame) + 1
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(trigger) loading data")
lines_Y = sc.textFile(inputFile_Y)
lines_x1 = sc.textFile(inputFile_x1)
lines_x2 = sc.textFile(inputFile_x2)
x1 = array([float(x) for x in lines_x1.collect()[0].split(' ')])
x2 = array([float(x) for x in lines_x2.collect()[0].split(' ')])
Y = lines_Y.map(parseVector).cache()
n = len(x1)

# compute sta
for lag in lags:
	logging.info('(trigger) getting triggered response at frame ' + str(lag))
	X = vstack((x1,x2))
	resp = Y.map(lambda y : getResp(y,X,lag))
	logging.info('(trigger) saving results...')
	nm = str(int(lag))
	savemat(outputFile+"/"+"resp-frame-"+nm+".mat",mdict={'resp':resp.collect()},oned_as='column',do_compression='true')

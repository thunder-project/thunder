import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(sta) usage: delay <master> <inputFile_X> <inputFile_y> <outputFile> <mxLag>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:])
	#ts = (ts - mean(ts))/std(ts)
	med = median(ts)
	ts = (ts - med) / (med)
	ts = (ts - mean(ts)) / std(ts)
	return ts

# parse inputs
sc = SparkContext(sys.argv[1], "delay")
inputFile_X = str(sys.argv[2])
inputFile_y = str(sys.argv[3])
mxLag = int(sys.argv[5])
outputFile = str(sys.argv[4]) + "-delay-mxLag-" + str(mxLag)
lags = arange(2*mxLag+1) - floor(2*mxLag/2)
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(delay) loading data")
lines_X = sc.textFile(inputFile_A)
lines_y = sc.textFile(inputFile_y)
y = array([float(x) for x in lines_y.collect()[0].split(' ')])
y = (y - mean(y))/std(y)
A = lines_A.map(parseVector)

# make predictor
Y = zeros((len(lags),len(y)))
for i in arange(len(lags)):
		lag = int(lags[i])
		tmp = roll(y,lag)
		if lag < 0:
			tmp[lag:] = 0
		if lag > 0:
			tmp[:lag] = 0
		Y[i,:] = tmp

# compute sta
logging.info('(delay) computing delay')
sta = A.map(lambda x : dot(Y,x))
norm = sta.map(lambda x : norm(x)).collect()

logging.info('(delay) saving results...')
savemat(outputFile+"/"+"norm.mat",mdict={'norm':norm},oned_as='column',do_compression='true')
for i in arange(len(lags)):
	filt = sta.map(lambda x : x[i]).collect()
	savemat(outputFile+"/"+"filt"+str(i)+".mat",mdict={'filt':filt},oned_as='column',do_compression='true')

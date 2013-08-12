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
  "(sta) usage: delay <master> <inputFile_A> <inputFile_y> <outputFile> <mxLag>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:])
	#ts = (ts - mean(ts))/std(ts)
	med = median(ts)
	ts = (ts - med) / (med)
	ts = (ts - mean(ts)) / std(ts)
	return ts

def getSta(x,y,lags):
	w = zeros((1,len(lags)))
	for i in arange(len(lags)):
		w[i] = mean(x * roll(y,int(lags[i])))
	return [dot(x,w)/sum(w),max(w)]

# parse inputs
sc = SparkContext(sys.argv[1], "delay")
inputFile_A = str(sys.argv[2])
inputFile_y = str(sys.argv[3])
mxLag = int(sys.argv[5])
outputFile = str(sys.argv[4]) + "-delay-mxLag-" + str(mxLag)
lags = arange(2*mxLag) - floor(2*mxLag/2)
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(delay) loading data")
lines_A = sc.textFile(inputFile_A)
lines_y = sc.textFile(inputFile_y)
y = array([float(x) for x in lines_y.collect()[0].split(' ')])
y = (y - mean(y))/std(y)
A = lines_A.map(parseVector).cache()

# compute sta
logging.info('(delay) computing delay')
sta = A.map(lambda x : getSta(x,y,lags))
logging.info('(delay) saving results...')
savemat(outputFile+"/"+"ph.mat",mdict={'ph':sta.map(lambda x : x[0]).collect()},oned_as='column',do_compression='true')
savemat(outputFile+"/"+"co.mat",mdict={'co':sta.map(lambda x : x[1]).collect()},oned_as='column',do_compression='true')

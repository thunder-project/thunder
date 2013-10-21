# localcorr <master> <inputFile_X> <outputFile> <neighborhood>"
# 
# correlate the time series for each pixel 
# in an image with its local neighborhood
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
  "(localcorr) usage: localcorr <master> <inputFile_X> <outputFile> <sz>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	meanVal = mean(ts)
	ts = (ts - meanVal) / (meanVal + 0.1) # convert to dff
	return ((int(vec[0]),int(vec[1]),int(vec[2])),ts) # (x,y,z),(tseries) pair 

def mapToNeighborhood((ind,ts),sz):
	rngX = range(-sz,sz+1,1)
	rngY = range(-sz,sz+1,1)
	out = list((ind,ts))
	for x in rngX :
		for y in rngY :
			if ~(x ==0 & y == 0) :
				newind = (ind[0] + x, ind[1] + y, ind[2])
				out.append((newind,ts))
	return out

# parse inputs
sc = SparkContext(sys.argv[1], "localcorr")
inputFile_X = str(sys.argv[2])
outputFile = str(sys.argv[3]) + "-localcorr"
sz = int(sys.argv[4])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(lowdim) loading data")
lines_X = sc.textFile(inputFile_X) # the data
X = lines_X.map(parseVector).cache()

# flatmap each time series to key value pairs where the key is now a neighborhood identifier and the value is the time series
neighbors = X.flatMap(lambda (k,v) : mapToNeighborhood((k,v),sz))

# reduceByKey to get the average time series
means = neighbors.reduceByKey(lambda x,y : x + y).map(lambda x : x / sz)

# join with the original data set and compute correlations
result = X.join(means).map(lambda (k,v) : corrcoef(v[0],v[1]))

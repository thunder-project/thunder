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
  "(localcorr) usage: localcorr <master> <inputFile_X> <outputFile> <sz> <mxX> <mxY>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	#meanVal = mean(ts)
	#ts = (ts - meanVal) / (meanVal + 0.1) # convert to dff
	return ((int(vec[0]),int(vec[1])),ts) # (x,y,z),(tseries) pair 

def clip(val,mn,mx):
	if (val < mn) : 
		val = mn
	if (val > mx) : 
		val = mx
	return val

def mapToNeighborhood(ind,ts,sz,mxX,mxY):
	# create a list of key value pairs with multiple shifted copies
	# of the time series ts
	rngX = range(-sz,sz+1,1)
	rngY = range(-sz,sz+1,1)
	out = list()
	for x in rngX :
		for y in rngY :
			newX = clip(ind[0] + x,1,mxX)
			newY = clip(ind[1] + y,1,mxY)
			newind = (newX, newY)
			out.append((newind,ts))
	return out

# parse inputs
sc = SparkContext(sys.argv[1], "localcorr")
inputFile_X = str(sys.argv[2])
outputFile = str(sys.argv[3]) + "-localcorr"
sz = int(sys.argv[4])
mxX = float(sys.argv[5])
mxY = float(sys.argv[6])

if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(lowdim) loading data")
lines_X = sc.textFile(inputFile_X) # the data
X = lines_X.map(parseVector).cache()

# flatmap each time series to key value pairs where the key is a neighborhood identifier and the value is the time series
neighbors = X.flatMap(lambda (k,v) : mapToNeighborhood(k,v,sz,mxX,mxY))

print(neighbors.first())

# # reduceByKey to get the average time series for each neighborhood
# means = neighbors.reduceByKey(lambda x,y : x + y,100).map(lambda (k,v) : (k, v / ((2*sz+1)**2)))

# print(means.first())

# join with the original time series data to compute correlations
# result = X.join(means)

# print(result.first())

# # get correlations
# corr = result.map(lambda (k,v) : mean(v[0]))

# # return keys because we'll need to sort on them post-hoc
# # TODO: use sortByKey once implemented in pyspark
# keys = result.map(lambda (k,v) : k)

# savemat(outputFile+"/"+"corr.mat",mdict={'corr':result.collect()},oned_as='column',do_compression='true')

# savemat(outputFile+"/"+"keys.mat",mdict={'keys':keys.collect()},oned_as='column',do_compression='true')
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

argsIn = sys.argv[1:]

if len(argsIn) < 6:
  print >> sys.stderr, \
  "(localcorr) usage: localcorr <master> <inputFile_X> <outputFile> <sz> <mxX> <mxY> <startInd> <endInd>"
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
			newind = (newX, newY, ind[2])
			out.append((newind,ts))
	return out

# parse inputs
sc = SparkContext(argsIn[0], "localcorr")
inputFile_X = str(argsIn[1])
outputFile = str(argsIn[2]) + "-localcorr"
sz = int(argsIn[3])
mxX = float(argsIn[4])
mxY = float(argsIn[5])

if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(lowdim) loading data")
lines_X = sc.textFile(inputFile_X) # the data

if len(argsIn) > 6 :
	logging.info("(lowdim) using specified indices")
	startInd = float(argsIn[6])
	endInd = float(argsIn[7])
	X = lines_X.map(lambda x : parseVector(x,"raw",1,(startInd,endInd))).cache()
else :
	X = lines_X.map(lambda x : parseVector(x,"raw",1)).cache()

# flatmap each time series to key value pairs where the key is a neighborhood identifier and the value is the time series
neighbors = X.flatMap(lambda (k,v) : mapToNeighborhood(k,v,sz,mxX,mxY))

print(neighbors.first())

# # reduceByKey to get the average time series for each neighborhood
means = neighbors.reduceByKey(lambda x,y : x + y).map(lambda (k,v) : (k, v / ((2*sz+1)**2)))

# join with the original time series data to compute correlations
result = X.join(means)

# get correlations and keys
# TODO: use sortByKey to avoid returning keys once implemented in pyspark
corr = result.map(lambda (k,v) : (k,corrcoef(v[0],v[1])[0,1]))

savemat(outputFile+"/"+"corr.mat",mdict={'corr':corr.collect()},oned_as='column',do_compression='true')

# savemat(outputFile+"/"+"keys.mat",mdict={'keys':keys.collect()},oned_as='column',do_compression='true')
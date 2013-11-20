# localcorr <master> <inputFile_X> <outputFile> <neighborhood> <mxX> <mxY>"
# 
# correlate the time series of each pixel 
# in an image with its local neighborhood
#

import sys
import os
from numpy import *
from scipy.linalg import *
from thunder.util.dataio import *
from thunder.util.signal import *
from thunder.util.image import *
from pyspark import SparkContext

argsIn = sys.argv[1:]

if len(argsIn) < 6:
  print >> sys.stderr, \
  "(localcorr) usage: localcorr <master> <dataFile> <outputFile> <sz> <mxX> <mxY>"
  exit(-1)

# parse inputs
sc = SparkContext(argsIn[0], "localcorr")
dataFile = str(argsIn[1])
outputFile = str(argsIn[2]) + "-localcorr"
sz = int(argsIn[3])
mxX = float(argsIn[4])
mxY = float(argsIn[5])
if not os.path.exists(outputFile) : os.makedirs(outputFile)

# parse data
data = sc.textFile(dataFile).map(lambda x : parse(x,"raw","xyz")).cache()

# flatmap each time series to key value pairs where the key is a neighborhood identifier and the value is the time series
neighbors = data.flatMap(lambda (k,v) : mapToNeighborhood(k,v,sz,mxX,mxY))
print(neighbors.first()) # for some reason unless i print here it gets stuck later

# reduceByKey to get the average time series for each neighborhood
means = neighbors.reduceByKey(lambda x,y : x + y).map(lambda (k,v) : (k, v / ((2*sz+1)**2)))

# join with the original time series data to compute correlations
result = data.join(means)

# get correlations and keys
# TODO: use sortByKey to avoid returning keys once implemented in pyspark
corr = result.map(lambda (k,v) : (k,corrcoef(v[0],v[1])[0,1]))

saveout(corr,outputFile,"corr","matlab")

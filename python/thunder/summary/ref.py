# ref <master> <dataFile> <outputDir> <mode>
# 
# compute summary statistics
#

import sys
import os
from numpy import *
from thunder.util.dataio import parseVector
from pyspark import SparkContext

argsIn = sys.argv[1:]


# def parse(line, filter="raw", inds=None):

# 	vec = [float(x) for x in line.split(' ')]
# 	ts = array(vec[3:]) # get tseries
# 	if filter == "dff" : # convert to dff
# 		meanVal = mean(ts)
# 		ts = (ts - meanVal) / (meanVal + 0.1)
# 	if inds is not None :
# 		if inds == "xyz" :
# 			return ((int(vec[0]),int(vec[1]),int(vec[2])),ts)
# 		if inds == "linear" :
# 			k = int(vec[0]) + int((vec[1] - 1)*1650)
# 			return (k,ts)
# 	else :
# 		return ts

if len(argsIn) < 4:
  print >> sys.stderr, \
  "(ref) usage: ref <master> <dataFile> <outputDir> <mode>"
  exit(-1)

# parse inputs
sc = SparkContext(argsIn[0], "ref")
dataFile = str(argsIn[1])
outputDir = str(argsIn[2])
mode = str(argsIn[3])
if not os.path.exists(outputDir) : os.makedirs(outputDir)

# parse data
lines = sc.textFile(dataFile)
data = lines.map(lambda x : parseVector(x,"raw","xyz")).cache()

# get z ordering
zinds = data.filter(lambda (k,x) : (k[0] == 1) & (k[1] == 1)).map(lambda (k,x) : k[2])
print(zinds.collect())
#saveout(zinds,outputDir,"zinds","matlab")

# compute ref
if mode == 'med':
	ref = data.map(lambda (k,x) : median(x))
if mode == 'mean':
	ref = data.map(lambda (k,x) : mean(x))
if mode == 'std':
	ref = data.map(lambda (k,x) : std((x - mean(x))/(mean(x)+0.1)))
if mode == 'perc':
	ref = data.map(lambda (k,x) : percentile(x,90))

#saveout(ref,outputDir,"ref"+mode,"matlab")

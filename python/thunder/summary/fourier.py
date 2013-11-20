# fourier <master> <inputFile> <outputFile> <freq>
# 
# computes the amplitude and phase of time series data
#

import sys
import os
from pyspark import SparkContext
from thunder.util.dataio import *
from thunder.util.signal import *

if len(sys.argv) < 5:
  print >> sys.stderr, \
  "(fourier) usage: fourier <master> <inputFile> <outputFile> <freq>"
  exit(-1)

def parse(line, filter="raw", inds=None):

	vec = [float(x) for x in line.split(' ')]
	ts = (vec[3:]) # get tseries
	if filter == "dff" : # convert to dff
		meanVal = mean(ts)
		ts = (ts - meanVal) / (meanVal + 0.1)
	if inds is not None :
		if inds == "xyz" :
			return ((int(vec[0]),int(vec[1]),int(vec[2])),ts)
		if inds == "linear" :
			k = int(vec[0]) + int((vec[1] - 1)*1650)
			return (k,ts)
	else :
		return ts

# parse inputs
sc = SparkContext(sys.argv[1], "fourier")
inputFile = str(sys.argv[2])
freq = int(sys.argv[4])
outputFile = str(sys.argv[3])+"-fourier"
if not os.path.exists(outputFile) : os.makedirs(outputFile)

# load data
data = sc.textFile(inputFile).map(lambda y : parse(y,"dff")).cache()

# do fourier on each time series
out = data.map(lambda x : getFourier(x,freq))

# save results
co = out.map(lambda x : x[0])
ph = out.map(lambda x : x[1])
print(co.first())

#saveout(co,outputFile,"co","matlab")
#saveout(ph,outputFile,"ph","matlab")

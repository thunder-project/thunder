# fourier <master> <dataFile> <outputFile> <freq>
# 
# computes the amplitude and phase of time series data
#

import sys
import os
from numpy import *
from numpy.fft import *
from pyspark import SparkContext
from thunder.util.dataio import *

if len(sys.argv) < 5:
  print >> sys.stderr, \
  "(fourier) usage: fourier <master> <dataFile> <outputFile> <freq>"
  exit(-1)

def parseVector(line, filter="raw", inds=None) :
		vec = [float(x) for x in line.split(' ')]
		ts = array(vec[3:]) # get tseries
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

def getFourier(vec,freq):
	vec = vec - mean(vec)
	nframes = len(vec)
	ft = fft(vec)
	ft = ft[0:int(fix(nframes/2))]
	ampFT = 2*abs(ft)/nframes;
	amp = ampFT[freq]
	co = zeros(size(amp));
	sumAmp = sqrt(sum(ampFT**2))
	co = amp / sumAmp
	ph = -(pi/2) - angle(ft[freq])
	if ph<0:
		ph = ph+pi*2
	return array([co,ph])

# parse inputs
sc = SparkContext(sys.argv[1], "fourier")
dataFile = str(sys.argv[2])
freq = int(sys.argv[4])
outputFile = str(sys.argv[3])+"-fourier"
if not os.path.exists(outputFile) : os.makedirs(outputFile)

# load data
lines = sc.textFile(dataFile)
#data = lines.map(lambda x : parseVector(x,"dff")).cache()
data = parse(lines, "dff").cache()

print(data.first())

# do fourier on each time series
out = data.map(lambda x : getFourier(x,freq))

# save results
co = out.map(lambda x : x[0])
ph = out.map(lambda x : x[1])

saveout(co,outputFile,"co","matlab")
saveout(ph,outputFile,"ph","matlab")

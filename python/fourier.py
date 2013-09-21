# fourier <master> <inputFile> <outputFile> <freq>
# 
# computes the amplitude and phase of each time series
#

import sys
import os
from numpy import *
from numpy.fft import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 5:
  print >> sys.stderr, \
  "(fourier) usage: fourier <master> <inputFile> <outputFile> <freq>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	med = median(ts)
	ts = (ts - med) / (med + 0.1) # convert to dff
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
inputFile = str(sys.argv[2])
freq = int(sys.argv[4])
outputFile = str(sys.argv[3])+"-fourier"
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# load data
logging.info('(fourier) loading data')
lines = sc.textFile(inputFile)
data = lines.map(parseVector)

# do fourier on each time series
logging.info('(fourier) doing fourier analysis')
out = data.map(lambda x : getFourier(x,freq))

# save results
logging.info('(fourier) saving results')
co = out.map(lambda x : x[0])
ph = out.map(lambda x : x[1])
savemat(outputFile+"/"+"co-freq-"+str(freq)+".mat",mdict={'co':co.collect()},oned_as='column',do_compression='true')
savemat(outputFile+"/"+"ph-freq-"+str(freq)+".mat",mdict={'ph':ph.collect()},oned_as='column',do_compression='true')
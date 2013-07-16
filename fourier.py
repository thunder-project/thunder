# fourier <master> <inputFile> <outputFile> <slices> <freq>
# 
# computes the amplitude and phase of each time series
# (stored as rows of a text file)
#

import sys
import os
from numpy import *
from numpy.fft import *
from pyspark import SparkContext

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(ica) usage: cca <master> <inputFile> <outputFile> <slices> <freq>"
  exit(-1)

def parseVector(line):
  return array([float(x) for x in line.split(' ')])

def getFourier(vec,freq):
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
sc = SparkContext(sys.argv[1], "cca")
inputFile = str(sys.argv[2]);
outputFile = str(sys.argv[3])
slices = int(sys.argv[4])
freq = int(sys.argv[5])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)

# do fourier on each time series
lines = sc.textFile(inputFile)
data = lines.map(parseVector)
out = data.map(lambda x : getFourier(x,freq)).collect()

savetxt(outputFile+"/"+"out-co-ph-freq-"+str(freq)+".txt",out,fmt='%.8f')


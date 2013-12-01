# fourier <master> <dataFile> <outputFile> <freq>
# 
# computes the amplitude and phase of time series data
#
# example:
# fourier.py local data/fish.txt results 12
#

import sys
import os
from numpy import mean, array, angle, abs, sqrt, zeros, fix, size, pi
from numpy.fft import fft
from thunder.util.dataio import *
from pyspark import SparkContext

argsIn = sys.argv[1:]
if len(argsIn) < 4:
    print >> sys.stderr, "(fourier) usage: fourier <master> <dataFile> <outputFile> <freq>"
    exit(-1)

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
sc = SparkContext(argsIn[0], "fourier")
dataFile = str(argsIn[1])
outputFile = str(argsIn[2]) + "-fourier"
freq = int(argsIn[3])
if not os.path.exists(outputFile) : os.makedirs(outputFile)

# load data
lines = sc.textFile(dataFile)
data = parse(lines, "dff")

# do fourier on each time series
out = data.map(lambda x : getFourier(x,freq)).cache()

# save results
co = out.map(lambda x : x[0])
ph = out.map(lambda x : x[1])

saveout(co,outputFile,"co","matlab")
saveout(ph,outputFile,"ph","matlab")

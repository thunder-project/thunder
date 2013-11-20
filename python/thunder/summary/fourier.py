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
saveout(co,outputFile,"co","matlab")
saveout(ph,outputFile,"ph","matlab")

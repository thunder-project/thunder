import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 4:
  print >> sys.stderr, \
  "(sta) usage: ref <master> <inputFile_X> <outputFile>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = ((int(vec[0]),int(vec[1]),int(vec[2])),array(vec[3:]))
	return ts

# parse inputs
sc = SparkContext(sys.argv[1], "ref")
inputFile_X = str(sys.argv[2])
outputFile = str(sys.argv[3])

# parse data
logging.info("(ref) loading data")
lines_X = sc.textFile(inputFile_X)
X = lines_X.map(parseVector)

# get z ordering
zvals = X.filter(lambda (k,d) : k[0] == 1 & k[1] == 1).map(lambda (k,d) : k[2]).collect()

# compute ref
logging.info('(ref) computing reference image')
ref = X.mapValues(lambda x : median(x))
logging.info('(ref) saving results...')
savemat(outputFile+"ref.mat",mdict={'ref':ref.collect()},oned_as='column',do_compression='true')
#savetxt(outputFile+"/"+"sta-lag-"+nm+".txt",sta.collect(),fmt='%.4f')

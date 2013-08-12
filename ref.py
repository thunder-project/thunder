import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 3:
  print >> sys.stderr, \
  "(sta) usage: ref <master> <inputFile_Y> <outputFile>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[1:])
	return (int(vec[0]),ts)

# parse inputs
sc = SparkContext(sys.argv[1], "ref")
inputFile_Y = str(sys.argv[2])
outputFile = str(sys.argv[3])

# parse data
logging.info("(ref) loading data")
lines_Y = sc.textFile(inputFile_Y)
Y = lines_Y.map(parseVector).cache()

# compute ref
logging.info('(ref) computing reference image')
ref = Y.map(lambda (k,x) : median(x))
logging.info('(ref) saving results...')
savemat(outputFile+"-ref.mat",mdict={'ref':ref.collect()},oned_as='column',do_compression='true')
#savetxt(outputFile+"/"+"sta-lag-"+nm+".txt",sta.collect(),fmt='%.4f')

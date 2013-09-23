# ref <master> <inputFile> <inds> <outputFile>
# 
# compute a reference image of an xyz stack
# each row is (x,y,z,time series)
#

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
  "(ref) usage: ref <master> <inputFile> <outputFile>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	return ((int(vec[0]),int(vec[1]),int(vec[2])),ts) # (x,y,z),(tseries) pair 

# parse inputs
sc = SparkContext(sys.argv[1], "ref")
inputFile = str(sys.argv[2])
outputFile = str(sys.argv[3])
logging.basicConfig(filename=outputFile+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(ref) loading data")
lines = sc.textFile(inputFile)
X = lines.map(parseVector)

# get z ordering
logging.info("(ref) getting z ordering")
zinds = X.filter(lambda (k,x) : (k[0] == 1000) & (k[1] == 1000)).map(lambda (k,x) : k[2]).collect()
savemat(outputFile+"zinds.mat",mdict={'zinds':zinds},oned_as='column',do_compression='true')

# compute ref
logging.info('(ref) computing reference image')
ref = X.map(lambda (k,x) : median(x)).collect()
logging.info('(ref) saving results...')
savemat(outputFile+"ref.mat",mdict={'ref':ref},oned_as='column',do_compression='true')
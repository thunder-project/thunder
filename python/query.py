import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 4:
  print >> sys.stderr, \
  "(query) usage: query <master> <inputFile_X> <inds> <outputFile>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	k = (vec(0) + (vec(1) - 1)*1025 + (vec(2) - 1)*1025*2000).toInt
	med = median(ts)
	ts = (ts - med) / (med + 0.1) # convert to dff
	return (k,ts)

# parse inputs
sc = SparkContext(sys.argv[1], "query")
inputFile_X = str(sys.argv[2])
indsFile = str(sys.argv[3])
outputFile = str(sys.argv[4]) + "-query"
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

logging.info("(query) loading data")
lines_X = sc.textFile(inputFile_X) # the data
X = lines_X.map(parseVector)

inds = loadmat(indsFile)['inds']
n = len(inds)

ts = data.filter(lambda (k,x) : k in inds).map(lambda (k,x) : x).reduce(lambda x,y :x+y) / n

savemat(outputFile+"ts.mat",mdict={'ts':ts},oned_as='column',do_compression='true')



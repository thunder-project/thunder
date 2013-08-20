import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(lowdim) usage: lowdim <master> <inputFile_X> <inputFile_t> <outputFile> <k>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	med = median(ts)
	ts = (ts - med) / (med) # convert to dff
	return ts

# parse inputs
sc = SparkContext(sys.argv[1], "lowdim")
inputFile_X = str(sys.argv[2])
inputFile_t = str(sys.argv[3])
outputFile = str(sys.argv[4]) + "-lowdim"
k = int(sys.argv[5])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(lowdim) loading data")
lines_X = sc.textFile(inputFile_X) # the data
X = lines_X.map(parseVector)
t = loadmat(inputFile_t)['trigInds'] # the triggers

out = resp.map(lambda x : mean(x))
savemat(outputFile+"/"+"scores-"+str(ik)+".mat",mdict={'scores':out.collect()},oned_as='column',do_compression='true')

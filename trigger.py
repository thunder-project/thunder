import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 7:
  print >> sys.stderr, \
  "(trigger) usage: trigger <master> <inputFile_X> <inputFile_t> <outputFile>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	return ((int(vec[0]),int(vec[1]),int(vec[2])),ts) # (x,y,z),(tseries) pair 

# parse inputs
sc = SparkContext(sys.argv[1], "trigger")
inputFile_X = str(sys.argv[2])
inputFile_t = str(sys.argv[3])
outputFile = str(sys.argv[4]) + "-trigger"
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(trigger) loading data")
lines_X = sc.textFile(inputFile_X) # the data
X = lines_X.map(parseVector).cache()
t = loadmat(inputFile_t)['trigInds'][0] # the triggers
n = len(t)

# get z ordering
zvals = X.filter(lambda (k,d) : k[0] == 1 & k[1] == 1).map(lambda (k,d) : k[2]).collect()
savemat(outputFile+"/"+"zinds.mat",mdict={'zinds':resp.collect()},oned_as='column',do_compression='true')

# compute triggered movie
for it in unique(t)
	logging.info('(trigger) getting triggered response at frame ' + str(it))
	resp = X.map(lambda x : mean(x[t==it])).collect()
	logging.info('(trigger) saving results...')
	savemat(outputFile+"/"+"resp-frame-"+str(int(it))+".mat",mdict={'resp':resp},oned_as='column',do_compression='true')

# ref <master> <inputFile> <inds> <outputFile>
# 
# compute summary statistics on an xyz stack
# each row is (x,y,z,timeseries)
#

import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from scipy.io import * 
from pyspark import SparkContext
import logging


argsIn = sys.argv[1:]

if len(argsIn) < 4:
  print >> sys.stderr, \
  "(ref) usage: ref <master> <inputFileX> <outputFile> <mode> <startInd> <endInd>"
  exit(-1)

def parseVector(line,mode="raw",xyz=0,inds=None):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	if inds is not None :
		ts = ts[inds[0]:inds[1]]
	if mode == "dff" :
		meanVal = mean(ts)
		ts = (ts - meanVal) / (meanVal + 0.1)
	if xyz == 1 :
		return ((int(vec[0]),int(vec[1]),int(vec[2])),ts)
	else :
		return ts

# parse inputs
sc = SparkContext(argsIn[0], "ref")
inputFile_X = str(argsIn[1])
outputFile = str(argsIn[2])
mode = str(argsIn[3])
logging.basicConfig(filename=outputFile+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(ref) loading data")
lines_X = sc.textFile(inputFile_X) # the data

if len(argsIn) > 4 :
	logging.info("(ref) using specified indices")
	startInd = float(argsIn[4])
	endInd = float(argsIn[5])
	X = lines_X.map(lambda x : parseVector(x,"raw",1,(startInd,endInd))).cache()
else :
	X = lines_X.map(lambda x : parseVector(x,"raw",1)).cache()

# get z ordering
logging.info("(ref) getting z ordering")
zinds = X.filter(lambda (k,x) : (k[0] == 1) & (k[1] == 1)).map(lambda (k,x) : k[2]).collect()
savemat(outputFile+"zinds.mat",mdict={'zinds':zinds},oned_as='column',do_compression='true')

# compute ref
logging.info('(ref) computing reference image')
if mode == 'med':
	ref = X.map(lambda (k,x) : median(x)).collect()
if mode == 'mean':
	ref = X.map(lambda (k,x) : mean(x)).collect()
if mode == 'std':
	ref = X.map(lambda (k,x) : std((x - mean(x))/(mean(x)+0.1))).collect()
if mode == 'perc':
	ref = X.map(lambda (k,x) : percentile(x,90)).collect()
logging.info('(ref) saving results...')
savemat(outputFile+mode+".mat",mdict={'ref':ref},oned_as='column',do_compression='true')
import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(sta) usage: sta <master> <inputFile_A> <inputFile_y> <outputFile> <mxLag>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[1:])
	ts = (ts - mean(ts))/std(ts)
	return ts

# parse inputs
sc = SparkContext(sys.argv[1], "sta")
inputFile_A = str(sys.argv[2])
inputFile_y = str(sys.argv[3])
mxLag = int(sys.argv[5])
outputFile = str(sys.argv[4]) + "-sta-mxLag-" + str(mxLag)
lags = arange(2*mxLag) - floor(2*mxLag/2)
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(sta) loading data")
lines_A = sc.textFile(inputFile_A)
lines_y = sc.textFile(inputFile_y)
y = array([float(x) for x in lines_y.collect()[0].split(' ')])
y = (y - mean(y))/std(y)
A = lines_A.map(parseVector).cache()

# compute sta
for lag in lags:
	logging.info('(sta) computing sta with time lag ' + str(lag))
	sta = sub.map(lambda x : mean(x * roll(y,int(lag))))
	logging.info('(sta) saving results...')
	nm = str(int(lag))
	if (lag < 0):
		nm = "n" + nm[1:]
	savemat(outputFile+"/"+"sta-lag-"+nm+".mat",mdict={'sta':sta.collect()},oned_as='column',do_compression='true')
	#savetxt(outputFile+"/"+"sta-lag-"+nm+".txt",sta.collect(),fmt='%.4f')

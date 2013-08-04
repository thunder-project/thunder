import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(regress) usage: regress <master> <inputFile_A> <inputFile_y1> <inputFile_y2> <outputFile> <mxLag>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[1:])
	ts = (ts - mean(ts))/std(ts)
	return ts

# parse inputs
sc = SparkContext(sys.argv[1], "sta")
inputFile_A = str(sys.argv[2])
inputFile_y1 = str(sys.argv[3])
inputFile_y2 = str(sys.argv[4])
mxLag = int(sys.argv[6])
outputFile = str(sys.argv[5]) + "-sta-mxLag-" + str(mxLag)
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(sta) loading data")
lines_A = sc.textFile(inputFile_A)
lines_y1 = sc.textFile(inputFile_y1)
lines_y2 = sc.textFile(inputFile_y2)
y1 = array([float(x) for x in lines_y1.collect()[0].split(' ')])
y2 = array([float(x) for x in lines_y2.collect()[0].split(' ')])
A = lines_A.map(parseVector)
d = A.count()
n = len(A.first())

# subtract the mean and cache
logging.info("(sta) subtracting mean")
meanVec = A.reduce(lambda x,y : x+y) / n
sub = A.map(lambda x : x - meanVec).cache()

# compute sta
for lag in lags:
	logging.info('(sta) computing sta with time lag ' + str(lag))
	sta = sub.map(lambda x : mean(x * y))
	logging.info('(sta) saving results...')
	nm = str(int(lag))
	if (lag < 0):
		nm = "n" + nm[1:]
	savetxt(outputFile+"/"+"sta-lag-"+nm+".txt",sta.collect(),fmt='%.4f')

# corr <master> <inputFile_A> <inputFile_y> <outputFile> <mxLag>
# 
# 


import sys
import os
from numpy import *
from scipy.stats.stats import pearsonr
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(corr) usage: corr <master> <inputFile_A> <inputFile_y> <outputFile> <mxLag>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[1:])
	ts = (ts - mean(ts))/std(ts)
	return (int(vec[0]),ts)

def correlate(x,y):
	r,p = pearsonr(x,y)
	return r

# parse inputs
sc = SparkContext(sys.argv[1], "corr")
inputFile_A = str(sys.argv[2])
inputFile_y = str(sys.argv[3])
mxLag = int(sys.argv[5])
outputFile = str(sys.argv[4]) + "-corr-mxLag-" + str(mxLag)
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(corr) loading data")
lines_A = sc.textFile(inputFile_A)
lines_y = sc.textFile(inputFile_y)
y = array([float(x) for x in lines_y.collect()[0].split(' ')])
y = (y - mean(y))/std(y)
A = lines_A.map(parseVector).cache()
d = A.count()
n = len(A.first()[1])

lags = arange(2*mxLag) - floor(2*mxLag/2)

# compute correlations
for lag in lags:
	logging.info('(corr) computing correlation with time lag ' + str(lag))
	out = A.map(lambda (k,x) : correlate(x,roll(y,int(lag))))
	logging.info('(corr) saving results')
	nm = str(int(lag))
	if (lag < 0):
		nm = "n" + nm[1:]
	savetxt(outputFile+"/"+"corr-lag-"+nm+".txt",out.collect(),fmt='%.4f')
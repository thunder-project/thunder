import sys
import os
from numpy import *
from scipy.io import * 
from pyspark import SparkContext
import logging


if len(sys.argv) < 4:
  print >> sys.stderr, \
  "(sta) usage: refBlock <master> <inputDir> <outputDir>"
  exit(-1)

def loadFile(index,iterator):
	nCols = 2034
	nRows = 1134
	nTime = 3430
	plane = int(floor(index/nCols))+1
	col = int(mod(index,nCols))
	inputfilename = inputDir + '/Plane{:02g}.stack'.format(plane)
	fid = open(inputfilename, 'rb')
	data = zeros((nRows,nTime))
	fid.seek(col*nRows*2,1)
	for it in range(nTime):
		data[:,it] = fromfile(fid,int16,nRows)
		fid.seek((nCols-1)*nRows*2,1)
	fid.close()
	yield data

# parse inputs
sc = SparkContext(sys.argv[1], "refBlock")
inputDir = str(sys.argv[2])
outputFile = str(sys.argv[3])
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

nPlanes = 41 # number of planes
nCols = 2034 # number of columns per plane
nRows = 1134 # number of rows per plane
X = sc.parallelize(range(nPlanes*nCols),nPlanes*nCols).mapPartitionsWithSplit(loadFile)
ref = concatenate(X.map(lambda x : median(x,axis=1)).collect())
savemat(outputFile+"ref.mat",mdict={'ref':ref},oned_as='column',do_compression='true')


# mean across time for all pixels
# concatenate(data.map(lambda x : sum(x,axis=1)).collect())

# mean across pixels for all time points
# data.map(lambda x : mean(x,axis=1)).reduce(lambda x,y : x+y)

# covariance matrix
# print(data.map(lambda x : dot(transpose(x),x)).reduce(lambda x,y : x+y) / (nPlanes * nRows * nCols)) 



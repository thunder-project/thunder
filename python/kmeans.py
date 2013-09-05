import sys
import os
from numpy import *
from scipy.io import * 
from pyspark import SparkContext
import logging

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries, drop x,y,z coords
	med = median(ts)
	ts = (ts - med) / (med) # convert to dff
	return ts

def closestPoint(p, centers):
  bestIndex = 0
  closest = float("+inf")
  for i in range(len(centers)):
    tempDist = sum((p - centers[i]) ** 2)
    if tempDist < closest:
      closest = tempDist
      bestIndex = i
  return bestIndex

if len(sys.argv) < 5:
  print >> sys.stderr, \
  "(kmeans) usage: kmeans <master> <inputFile> <outputFile> <k>"
  exit(-1)

sc = SparkContext(sys.argv[1], "kmeans")
inputFile_X = str(sys.argv[2])
k = int(sys.argv[4])
outputFile = str(sys.argv[3]) + "-kmeans-" + str(k)
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

logging.info("(kmeans) loading data")
lines_X = sc.textFile(inputFile_X)
X = lines_X.map(parseVector).cache()

kPoints = X.take(k)

convergeDist = 0.001
tempDist = 1.0
iteration = 0

while tempDist > convergeDist:
	logging.info("(kmeans) starting iteration " + str(iteration))
	closest = X.map(
	    lambda p : (closestPoint(p, kPoints), (p, 1)))
	pointStats = closest.reduceByKey(
	    lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2))
	newPoints = pointStats.map(
	    lambda (x, (y, z)): (x, y / z)).collect()

	tempDist = sum(sum((kPoints[x] - y) ** 2) for (x, y) in newPoints)

	for (x, y) in newPoints:
	    kPoints[x] = y

	iteration = iteration + 1

logging.info("(kmeans) saving results")
labels = X.map( lambda p : closestPoint(p, kPoints)).collect()
savemat(outputFile+"/"+"labels.mat",mdict={'labels':labels},oned_as='column',do_compression='true')
savemat(outputFile+"/"+"centers.mat",mdict={'kPoints':kPoints},oned_as='column',do_compression='true')
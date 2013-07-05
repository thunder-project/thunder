# rpca <master> <inputFile> <outputFile> <slices> 

import sys
import os
from numpy import *
from scipy.linalg import *
from pyspark import SparkContext

if len(sys.argv) < 5:
  print >> sys.stderr, \
    "(rpca) usage: rpca <master> <inputFile> <outputFile> <slices>"
  exit(-1)

def parseVector(line):
    return array([float(x) for x in line.split(' ')])

def shrinkVec(x,thresh):
	tmp = abs(x)-thresh
	tmp[tmp<0] = 0
	return tmp

def svdThreshold(RDD,thresh):
	cov = RDD.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y))
	w, v = eig(cov)
	sw = sqrt(w)
	inds = (sw-thresh)>0
	d = (1/sw[inds]) * shrinkVec(sw[inds],thresh)
	vthresh = real(dot(dot(v[:,inds],diag(d)),transpose(v[:,inds])))
	return RDD.map(lambda x : dot(x,vthresh))

def shrinkage(RDD,thresh):
	return RDD.map(lambda x : sign(x) * shrinkVec(x,thresh))

# parse inputs
sc = SparkContext(sys.argv[1], "rpca")
inputFile = str(sys.argv[2])
outputFile = str(sys.argv[3])
slices = int(sys.argv[4])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)

# load data
lines = sc.textFile(inputFile,slices)
data = lines.map(parseVector)
n = data.count()
m = len(data.first())

# mean subtraction
#meanVec = data.reduce(lambda x,y : x+y) / n
#sub = data.map(lambda x : x - meanVec)

# create broadcast variables
print "(rpca) broadcasting variables"
M = array(data.collect())
L = zeros((n,m))
S = zeros((n,m))
Y = zeros((n,m))

mu = float(12)
lam = 1/sqrt(n)

iterNum = 0
iterMax = 50

while iterNum < iterMax:
	print "(rpca) starting iteration " + str(iterNum)
	iterNum += 1
	MSY = sc.parallelize(M - S + (1/mu)*Y).cache()
	L = svdThreshold(MSY,1/mu).collect()
	MLY = sc.parallelize(M - L + (1/mu)*Y)
	S = shrinkage(MLY,lam/mu).collect()
	Y = Y + mu*(M - L - S)

savetxt(outputFile+"/"+"out-L-"+outputFile+".txt",L,fmt='%.8f')
savetxt(outputFile+"/"+"out-S-"+outputFile+".txt",S,fmt='%.8f')

